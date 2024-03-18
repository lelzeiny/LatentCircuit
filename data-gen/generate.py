import utils
import algorithms
import torch
import hydra
from omegaconf import OmegaConf, open_dict
import common
import os
import time

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.join(cfg.out_dir, f"{cfg.dataset_name}.{cfg.seed}")
    checkpointer = common.Checkpointer(os.path.join(out_dir, "latest.ckpt"))
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass
    print(f"saving outputs to: {out_dir}")
    torch.manual_seed(cfg.seed)

    # Generator
    circuit_gen = algorithms.V1(**cfg.gen_params)

    # Prepare logger
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"circuit_gen.{cfg.dataset_name}.{cfg.seed}"
        outputs.append(common.logger.WandBOutput(wandb_run_name, cfg))
    
    step = common.Counter()
    logger = common.Logger(step, outputs)
    utils.save_cfg(cfg, os.path.join(out_dir, "config.yaml"))
    
    checkpointer.register({
        "step": step,
    })

    # Start training
    print(OmegaConf.to_yaml(cfg))
    print(f"==== Start Generation on Device: {device} ====")
    t_0 = time.time()
    t_1 = time.time()
    samples = []

    while step < cfg.num_samples:
        # generate data
        sample = circuit_gen.sample()
        samples.append(sample)
        # append to samples
        step.increment()
        # Generate number of terminals for each instance
        if (int(step)) % cfg.print_every == 0:
            t_2 = time.time()
            # TODO use something better than pickle
            utils.save_pickle(samples, os.path.join(out_dir, f"{int(step):08d}.pickle"))
            samples = []
            logger.add({
                "time_elapsed": t_2-t_0, 
                "ms_per_step": 1000*(t_2-t_1)/cfg.print_every
                })
            logger.write()
            checkpointer.save()
            t_1 = t_2
            

if __name__=="__main__":
    main()
