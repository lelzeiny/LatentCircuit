from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import numpy as np
import torch
import pickle

def save_cfg(cfg, path):
    with open(path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

def load_cfg(path):
    with open(path, "r") as f:
        return OmegaConf.load(f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def visualize(x, attr, mask = None):
    """ 
    Visualizes the X with node attributes, returning an numpy image
    x,y are floats normalized to canvas size (from -1 to 1)
    attr are also normalized to canvas size
    x: (V, 2) x,y position
    attr: (V, 2) sizes
    """
    width, height = 128, 128
    background_color = "white"
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    h_step = 1 / len(x)
    for i, (pos, shape) in enumerate(zip(x, attr)):
        left = pos[0]
        top = pos[1] + shape[1]
        right = pos[0] + shape[0]
        bottom = pos[1]
        inbounds = (left>=-1) and (top<=1) and (right<=1) and (bottom>=-1)

        left = (0.5 + left/2) * width
        right = (0.5 + right/2) * width
        top = (0.5 - top/2) * height
        bottom = (0.5 - bottom/2) * height

        color = hsv_to_rgb(i * h_step, 1 if (mask is None or not mask[i]) else 0.2, 0.9 if inbounds else 0.5)
        draw.rectangle([left, top, right, bottom], fill=color)

    return np.array(image)

def hsv_to_rgb(h, s, v):
    """
    Converts HSV (Hue, Saturation, Value) color space to RGB (Red, Green, Blue).
    h: float [0, 1] - Hue
    s: float [0, 1] - Saturation
    v: float [0, 1] - Value
    Returns: tuple (r, g, b) representing RGB values in the range [0, 255]
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255), int(g * 255), int(b * 255)

def debug_plot_img(x, name = "debug_img", autoscale = False):
    # x is (C, H, W) image, this function plots and saves to file
    # assumes images are [-1, 1]
    import matplotlib.pyplot as plt
    # scaling
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = x.movedim(0,-1)
    x = (x + 1)/2 if not autoscale else (x-x.min())/(x.max()-x.min())
    plt.imshow(x.cpu().detach().numpy())
    plt.savefig(name)