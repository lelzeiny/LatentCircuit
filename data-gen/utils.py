from omegaconf import OmegaConf

def save_cfg(cfg, path):
    with open(path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

def load_cfg(path):
    with open(path, "r") as f:
        return OmegaConf.load(f)