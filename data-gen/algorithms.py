from distributions import get_distribution
import torch

class V1:
    def __init__(self, max_instance, stop_density, aspect_ratio_dist, instance_size_dist, num_terminals_dist):
        self.max_instance = max_instance
        self.stop_density = stop_density
        self.aspect_ratio_dist = aspect_ratio_dist
        self.instance_size_dist = instance_size_dist
        self.num_terminals_dist = num_terminals_dist

    def sample(self):
        # Generate instance sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,))
        # TODO clip
        x_sizes = get_distribution(**self.instance_size_dist).sample((self.max_instance,))
        y_sizes = x_sizes * aspect_ratio

        # sample number of terminals
        num_terminals = get_distribution(**self.num_terminals_dist).sample((self.max_instance,)).int() # TODO condition dist on area of instance per Rent's
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(x_sizes, y_sizes, max_num_terminals)

        # place samples individually
        import ipdb; ipdb.set_trace()
        return 1
    
    def get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals):
        # x_sizes: (num_instances)
        # y_sizes: (num_instances)
        # max_num_terminals: int
        half_perim = (x_sizes + y_sizes)
        
        terminal_locations = get_distribution("uniform", {"low": 0, "high": half_perim}).sample((max_num_terminals,)) # (max_term, num_instances)
        terminal_flip = get_distribution("bernoulli", {"probs": 0.5}).sample((max_num_terminals, self.max_instance)) # (max_term, num_instances)
        terminal_flip = (2 * terminal_flip) - 1
        is_x = (terminal_locations > x_sizes).float()

        x_sizes = x_sizes.unsqueeze(dim=0)
        y_sizes = y_sizes.unsqueeze(dim=0)
        terminal_offset_x = torch.clamp(terminal_locations, torch.zeros_like(x_sizes), x_sizes) - (x_sizes/2) 
        terminal_offset_y = torch.clamp(terminal_locations-x_sizes, torch.zeros_like(y_sizes), y_sizes) - (y_sizes/2)

        terminal_offset_x = terminal_flip * terminal_offset_x
        terminal_offset_y = terminal_flip * terminal_offset_y

        terminal_offset = torch.stack((terminal_offset_x, terminal_offset_y), dim=-1).movedim(1, 0) # (num_instances, max_term, 2)
        return terminal_offset
