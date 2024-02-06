import utils
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import common
import os
import time

@hydra.main(version_base=None, config_path="configs", config_name="config_graph")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_dir = os.path.join(cfg.log_dir, f"{cfg.task}.{cfg.method}.{cfg.seed}")
    sample_dir = os.path.join(log_dir, "samples")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(sample_dir)
    except FileExistsError:
        pass
    print(f"saving checkpoints to: {log_dir}")
    torch.manual_seed(cfg.seed)

    # Preparing dataset
    train_set, val_set = utils.load_graph_data(cfg.task, augment = cfg.augment, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = train_set[0][0].shape
    dataloader = utils.GraphDataLoader(train_set, val_set, cfg.batch_size, cfg.val_batch_size, device)
    with open_dict(cfg):
        if cfg.family == "cond_diffusion":
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })
        else:
            raise NotImplementedError

    # Preparing model
    model_types = {"cond_diffusion": models.CondDiffusionModel}
    if cfg.implementation == "custom":
        model = model_types[cfg.family](**cfg.model).to(device)
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
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))

    # Load checkpoint if exists
    checkpointer.register({
        "step": step,
        "model": model,
        "optim": optim,
    })
    checkpointer.load(
        None if (cfg.from_checkpoint == "none" or cfg.from_checkpoint is None) 
        else os.path.join(cfg.log_dir, cfg.from_checkpoint)
    )
    
    # Start training
    print(OmegaConf.to_yaml(cfg)) 
    print(f"model has {num_params} params")
    print(f"==== Start Training on Device: {device} ====")
    model.train()

    t_0 = time.time()
    t_1 = time.time()
    best_loss = 1e12
    while step < cfg.train_steps:
        x, cond = dataloader.get_batch("train")
        # x has (B, N, 2); netlist_data is a single graph in tg.Data format
        t = torch.randint(1, cfg.model.max_diffusion_steps + 1, [x.shape[0]], device = device)
        optim.zero_grad()
        loss, model_metrics = model.loss(x, cond, t)
        loss.backward()
        optim.step()
        train_metrics.add({"loss": loss.cpu().item()})
        train_metrics.add(model_metrics)
        step.increment()

        if (int(step)) % cfg.print_every == 0:
            t_2 = time.time()
            x_val, cond_val = dataloader.get_batch("val")
            train_logs = utils.validate(x, model, cond)
            val_logs = utils.validate(x_val, model, cond_val)

            logger.add({
                "time_elapsed": t_2-t_0, 
                "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                })
            logger.add(train_metrics.result())
            logger.add(val_logs, prefix="val")
            logger.add(train_logs, prefix="train")

            # display example images
            for split in ["train", "val"]:
                x_disp, cond_disp = dataloader.get_display_batch(cfg.display_examples, split = split)
                utils.display_graph_samples(cfg.display_examples, x_disp, cond_disp, model, logger, prefix = split)
                utils.display_forward_graph_samples(x_disp, cond_disp, model, logger, prefix = split)
            logger.write()
            t_1 = t_2

            checkpointer.save() # save latest checkpoint
            if val_logs["loss"] < best_loss:
                best_loss = val_logs["loss"]
                checkpointer.save(os.path.join(log_dir, "best.ckpt"))
                print("saving best model")

        if (cfg.eval_every > 0) and (int(step)) % cfg.eval_every == 0:
            print("generating evaluation report")
            t3 = time.time()
            utils.generate_report(cfg.eval_samples, dataloader, model, logger, policy = cfg.eval_policy)
            checkpointer.save(os.path.join(log_dir, f"step_{int(step)}.ckpt"))
            logger.write()
            t4 = time.time()
            print(f"generated report in {t4-t3:.3f} sec")

    # output eval samples
    utils.save_outputs(val_set, model, cfg.num_output_samples, save_folder=sample_dir, output_number_offset=3300)

if __name__=="__main__":
    main()
