import utils
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import common
import os
import time

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass
    print(f"saving checkpoints to: {log_dir}")
    torch.manual_seed(cfg.seed)

    # Preparing dataset
    train_set, val_set, text_labels = utils.load_data(cfg.task, augment = cfg.augment, train_data_limit = cfg.data_limit)
    sample_shape = train_set[0][0].shape
    dataloader = utils.DataLoader(train_set, val_set, cfg.batch_size, cfg.val_batch_size, device)
    with open_dict(cfg):
        cfg.model.update({
            "num_classes": cfg.num_classes,
            "in_channels": sample_shape[0],
            "image_size": (sample_shape[1], sample_shape[2]),
            "device": device,
        })
    
    # Preparing model
    if cfg.implementation == "custom":
        model = models.DiffusionModel(**cfg.model).to(device)
    else:
        raise NotImplementedError
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_metrics = common.Metrics()

    # Prepare logger
    num_params = sum([param.numel() for param in model.parameters()])
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    step = common.Counter()
    logger = common.Logger(step, outputs)

    # Load checkpoint if exists
    checkpointer.register({
        "step": step,
        "model": model,
        "optim": optim,
    })
    checkpointer.load()
    
    # Start training
    print(OmegaConf.to_yaml(cfg)) 
    print(f"model has {num_params} params")
    print(f"==== Start Training on Device: {device} ====")
    model.train()

    t_0 = time.time()
    t_1 = time.time()
    best_loss = 1e12
    while step < cfg.train_steps:
        x, y = dataloader.get_batch("train")
        t = torch.randint(1, cfg.model.max_diffusion_steps + 1, [x.shape[0]], device = device) 
        # t = int(cfg.model.max_diffusion_steps) * torch.ones([x.shape[0]], device = device, dtype = torch.long) # TODO remove
        optim.zero_grad()
        loss, model_metrics = model.loss(x, t)
        loss.backward()
        optim.step()
        train_metrics.add({"loss": loss.cpu().item()})
        train_metrics.add(model_metrics)
        step.increment()

        if (int(step)) % cfg.print_every == 0:
            t_2 = time.time()
            x_val, y_val = dataloader.get_batch("val")
            train_logs = utils.validate(x, model)
            val_logs = utils.validate(x_val, model)

            logger.add({
                "time_elapsed": t_2-t_0, 
                "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                })
            logger.add(train_metrics.result())
            logger.add(val_logs, prefix="val")
            logger.add(train_logs, prefix="train")

            # display example images
            x_disp, y_disp = dataloader.get_display_batch(cfg.display_images)
            # utils.display_predictions(x_disp, y_disp, model, logger, prefix = "val", text_labels = text_labels)
            # utils.display_predictions(x[:cfg.display_images], y[:cfg.display_images], model, logger, prefix = "train", text_labels = text_labels)
            utils.display_samples(cfg.display_images, model, logger, prefix = "val")
            utils.display_forward_samples(x_disp, model, logger, prefix = "val")
            logger.write()
            t_1 = t_2

            checkpointer.save() # save latest checkpoint
            if val_logs["loss"] < best_loss:
                best_loss = val_logs["loss"]
                checkpointer.save(os.path.join(log_dir, "best.ckpt"))
                print("saving best model")

if __name__=="__main__":
    main()
