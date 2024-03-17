from distributions import get_distribution
import torch

class V1:
    def __init__(self, max_instance, aspect_ratio_dist, instance_size_dist):
        self.max_instance = max_instance
        self.aspect_ratio_dist = aspect_ratio_dist
        self.instance_size_dist = instance_size_dist

    def sample(self):
        # Generate instance sizes
        aspect_ratio = get_distribution((self.max_instance,), **self.aspect_ratio_dist)
        # TODO clip
        x_sizes = get_distribution((self.max_instance), **self.instance_size_dist)
        y_sizes = x_sizes * aspect_ratio
        areas = x_sizes * y_sizes
        import ipdb; ipdb.set_trace()

        return 1