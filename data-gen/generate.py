import utils
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

        # append to samples
        step.increment()

    # save outputs

if __name__=="__main__":
    main()
