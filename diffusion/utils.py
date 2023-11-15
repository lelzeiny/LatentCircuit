import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import wandb
import os
from torch.utils.data import Subset

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