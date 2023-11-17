import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import wandb
import os
import pathlib
from torch.utils.data import Subset
import pickle
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from PIL import Image, ImageDraw

@torch.no_grad()
def validate(x_val, model):
    model.eval()
    t = torch.randint(1, model.max_diffusion_steps + 1, [x_val.shape[0]], device = x_val.device)
    loss, model_metrics = model.loss(x_val, t)
    logs = {
        "loss": loss.cpu().item()
    }
    logs.update(model_metrics)
    model.train()
    return logs

@torch.no_grad()
def display_predictions(x_val, y_val, model, logger, prefix = "val", text_labels = None):
    model.eval()
    log_probs = model(x_val)
    probs = torch.nn.functional.softmax(log_probs, dim=-1)
    predictions = log_probs.argmax(dim=-1)
    for image, pred, label, prob, logit in zip(
        torch.movedim(x_val, 1, -1).cpu().numpy(), 
        predictions.cpu().numpy(), 
        y_val.cpu().numpy(),
        probs.cpu().numpy(),
        log_probs.cpu().numpy(),
        ):
        log_image = wandb.Image(image)
        logs = {
            "examples": {
                "image": log_image,
                "prediction": text_labels[pred] if text_labels else pred,
                "ground truth": text_labels[label] if text_labels else label,
                "pred prob": prob[pred],
                "truth prob": prob[label],
                "logit histogram": logit,
            },
        }
        logger.add(logs, prefix = prefix)
    model.train()

@torch.no_grad()
def display_samples(batch_size, model, logger, intermediate_every = 200, prefix = "val"):
    model.eval()
    samples, intermediates = model.reverse_samples(batch_size, intermediate_every = intermediate_every)
    intermediate_stats = compute_intermediate_stats(intermediates)
    intermediates = torch.cat(intermediates, dim = -1) # concat along width
    for idx, (image, intermediate_image) in enumerate(zip(
        torch.movedim(samples, 1, -1).cpu().numpy(),
        torch.movedim(intermediates, 1, -1).cpu().numpy()
    )):
        log_image = wandb.Image(image)
        log_intermediate = wandb.Image(intermediate_image)
        logs = {
            "reverse_examples": {
                "sample": log_image,
                "intermediates": log_intermediate,
            },
        }
        for stat_name, stat in intermediate_stats.items():
            logs["reverse_examples"][stat_name] = stat[idx]
        logger.add(logs, prefix = prefix)
    model.train()

@torch.no_grad()
def display_forward_samples(x_val, model, logger, intermediate_every = 200, prefix = "val"):
    model.eval()
    intermediates = model.forward_samples(x_val, intermediate_every = intermediate_every)
    intermediate_stats = compute_intermediate_stats(intermediates)
    intermediates = torch.cat(intermediates, dim = -1) # concat along width
    for idx, (image, intermediate_image) in enumerate(zip(
        torch.movedim(x_val, 1, -1).cpu().numpy(),
        torch.movedim(intermediates, 1, -1).cpu().numpy(),
    )):
        log_image = wandb.Image(image)
        log_intermediate = wandb.Image(intermediate_image)
        logs = {
            "forward_examples": {
                "image": log_image,
                "intermediates": log_intermediate,
            },
        }
        for stat_name, stat in intermediate_stats.items():
            logs["forward_examples"][stat_name] = stat[idx]
        logger.add(logs, prefix = prefix)
    model.train()

def compute_intermediate_stats(intermediates):
    # input: intermediates is a list, each is (B, C, H, W)
    # outputs: dict of stats, each value is torch tensor with shape (B, T)
    stats_to_compute = {"mean": torch.mean, "std": torch.std}
    stats = {}
    for stat_name, stat_fn in stats_to_compute.items():
        stat_list = [stat_fn(image.view(image.shape[0], -1), dim=1) for image in intermediates]
        stat = torch.cat(stat_list, -1)
        stats[stat_name] = stat
    return stats

def load_data(dataset_name, augment = False, train_data_limit = None):
    dataset_path = os.path.join(os.path.dirname(__file__), f'../datasets/vision/{dataset_name}')
    if dataset_name == "cifar10":
        transform_augment_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            ]
        transform_list = [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        transform_train = transforms.Compose((transform_augment_list + transform_list) if augment else transform_list)
        transform_val = transforms.Compose(transform_list)
        train_set = torchvision.datasets.CIFAR10(root = dataset_path, train=True, download=True, transform=transform_train)
        val_set = torchvision.datasets.CIFAR10(root = dataset_path, train=False, download=True, transform=transform_val)
        classes = train_set.classes
    elif dataset_name == "mnist":
        transform_augment_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
        ]
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
        transform_train = transforms.Compose((transform_augment_list + transform_list) if augment else transform_list)
        transform_val = transforms.Compose(transform_list)
        train_set = torchvision.datasets.MNIST(root = dataset_path, train=True, download=True, transform=transform_train)
        val_set = torchvision.datasets.MNIST(root = dataset_path, train=False, download=True, transform=transform_val)
        classes = None
    elif dataset_name == "inat-mini":
        transform_augment_list = [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomRotation(degrees=20),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.Resize(384),
            transforms.RandomCrop(384, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        transform_list = [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform_train = transforms.Compose((transform_augment_list + transform_list) if augment else transform_list)
        transform_val = transforms.Compose(transform_list)
        target_type = "class"
        train_set = torchvision.datasets.INaturalist(
            root = dataset_path, 
            version = "2021_train_mini",
            target_type = target_type,
            download = False, 
            transform = transform_train)
        val_set = torchvision.datasets.INaturalist(
            root = dataset_path,
            version = "2021_valid",
            target_type = target_type,
            download = False,
            transform = transform_val)
        classes = None
    elif dataset_name == "celeba":
        transform_augment_list = [
            transforms.RandomHorizontalFlip(),
            ]
        transform_list = [
            transforms.Resize((128, 128)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        transform_train = transforms.Compose((transform_augment_list + transform_list) if augment else transform_list)
        transform_val = transforms.Compose(transform_list)
        train_set = torchvision.datasets.CelebA(root = dataset_path, target_type="identity", split="train", download=True, transform=transform_train)
        val_set = torchvision.datasets.CelebA(root = dataset_path, target_type="identity", split="valid", download=True, transform=transform_val)
        classes = None    
    else:
        raise NotImplementedError
    if train_data_limit is not None and train_data_limit != "none":
        train_set = Subset(train_set, torch.arange(train_data_limit))
    return train_set, val_set, classes

def load_graph_data(dataset_name, augment = False, train_data_limit = None):
    dataset_path = os.path.join(os.path.dirname(__file__), f'../datasets/graph/{dataset_name}')
    if dataset_name == "placement-v0":
        chip_size = 250
        val_set = [] #TODO fix this
        train_cond_path = os.path.join(dataset_path, "graph0.pickle")
        train_x_path = os.path.join(dataset_path, "output0.pickle")
        train_cond = load_and_parse_graph(train_cond_path)
        train_x = torch.tensor(open_pickle(train_x_path) / 1000, dtype=torch.float32) # TODO fix this
        train_cond.x = (train_cond.x / (chip_size))
        train_x = 2 * (train_x / chip_size) - 1
        train_set = [(train_x, train_cond)] 
    else:
        raise NotImplementedError
    if train_data_limit is not None and train_data_limit != "none":
        train_set = Subset(train_set, torch.arange(train_data_limit))
    return train_set, val_set

class DataLoader:
    def __init__(
            self, 
            train_dataset, 
            val_dataset, 
            train_batch_size,
            val_batch_size,
            train_device = "cuda",
            num_workers = 8,
            pin_memory = False,
        ):
        self.device = train_device
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_set = train_dataset
        self.val_set = val_dataset
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, pin_memory_device=train_device if pin_memory else '')
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, pin_memory_device=train_device if pin_memory else '')
        self._get_train_batch = self._train_gen()
        self._get_val_batch = self._val_gen()
        self._display_x = None
        self._display_y = None

    def get_batch(self, split):
        assert split in ("train", "val"), "split argument has to be one of 'train' or 'val'"
        x, y = next(self._get_train_batch) if split == "train" else next(self._get_val_batch)
        return x.to(self.device), y.to(self.device)
    
    def get_display_batch(self, num_images):
        assert num_images <= self.val_batch_size, "num images must be smaller than batch size"
        if (self._display_x is None) or (self._display_y is None):
            x, y = self.get_batch("val")
            self._display_x = x[:num_images]
            self._display_y = y[:num_images]
        return self._display_x, self._display_y

    def _train_gen(self):
        while True:
            for data in self.train_loader:
                yield data
    
    def _val_gen(self):
        while True:
            for data in self.val_loader:
                yield data
    
    def get_train_size(self):
        return len(self.train_set)

    def get_val_size(self):
        return len(self.val_set)
    
class GraphDataLoader:
    def __init__(
            self, 
            train_dataset, 
            val_dataset, 
            train_batch_size,
            val_batch_size,
            train_device = "cuda",
            num_workers = 8,
            pin_memory = False,
        ):
        self.device = train_device
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_set = train_dataset
        self.val_set = val_dataset
    
    def get_batch(self, split):
        assert split in ("train", "val"), "split argument has to be one of 'train' or 'val'"
        dataset = self.train_set if split=="train" else self.val_set
        batch_size = self.train_batch_size if split=="train" else self.val_batch_size
        idx = torch.randint(0, len(dataset), [1], device = self.device) # TODO support larger batch sizes
        x, y = dataset[idx]
        return x.to(self.device).view(1, *x.shape).expand(batch_size, *x.shape), y.to(self.device)

    def get_train_size(self):
        return len(self.train_set)

    def get_val_size(self):
        return len(self.val_set)

def open_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_and_parse_graph(path):
    """loads networkx graph from pickle, gets Data object"""
    graph = open_pickle(path) #networkx graph
    attr_replace = {node[0]: {k:float(v) for k, v in node[1].items()} for node in graph.nodes(data=True)}
    nx.set_node_attributes(graph, attr_replace)    

    digraph = nx.DiGraph()

    for node in graph.nodes(data=True):
        digraph.add_node(node[0], **node[1])

    for edge in graph.edges(data=True):
        attr = edge[2]
        attr.pop('u_id')
        attr.pop('v_id')
        digraph.add_edge(edge[0], edge[1], **attr) # create the original edge
        temp = attr['u_pinx']
        attr['u_pinx'] = attr['v_pinx']
        attr['v_pinx'] = temp
        temp = attr['u_piny']
        attr['u_piny'] = attr['v_piny']
        attr['v_piny'] = temp
        digraph.add_edge(edge[1], edge[0], **attr) #create the flipped, duplicate edge
    return from_networkx(digraph, group_edge_attrs=all, group_node_attrs=all)

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

def visualize(X, attr):
    """ Visualizes the X with node attributes, returning an numpy image"""

    width, height = 128
    background_color = "white"

    image = Image.new("RGB", (width, height), background_color)

    draw = ImageDraw.Draw(image)

    h_step = 1 / len(X)
    

    #pos[0] = x
    #pos[1] = y
    #shape[0] = width
    #shape[1] = height
    for i, (pos, shape) in enumerate(zip(X, attr)):
        topLeft = (pos[0] - width / 2, pos[1] + height / 2)
        bottomRight = (pose[0] + width / 2, pose[1] - height / 2)
        color = hsv_to_rgb(i * h_step, 1, 1)
        draw.rectangle([topLeft, bottomRight], fill=color)

    return np.array(image)


