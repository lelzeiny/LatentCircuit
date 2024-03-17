from distributions import get_distribution
import utils
import torch
import shapely
import numpy as np

class V1:
    def __init__(
            self, 
            max_instance, 
            stop_density,
            max_attempts_per_instance,
            aspect_ratio_dist, 
            instance_size_dist, 
            num_terminals_dist,
            edge_dist,
            source_terminal_dist,
        ):
        self.max_instance = max_instance
        self.stop_density = stop_density
        self.aspect_ratio_dist = aspect_ratio_dist
        self.instance_size_dist = instance_size_dist
        self.num_terminals_dist = num_terminals_dist
        self.edge_dist = edge_dist
        self.max_attempts_per_instance = max_attempts_per_instance
        self.source_terminal_dist = source_terminal_dist

    def sample(self):
        # Generate instance sizes TODO use clipped poisson for sizes
        aspect_ratio = get_distribution(**self.aspect_ratio_dist).sample((self.max_instance,))
        long_size = get_distribution(**self.instance_size_dist).sample((self.max_instance,))
        short_size = aspect_ratio * long_size
        long_x = get_distribution("bernoulli", {"probs": 0.5}).sample((self.max_instance,))

        x_sizes = long_x * long_size + (1-long_x) * (short_size)
        y_sizes = (1-long_x) * long_size + (long_x) * (short_size)

        # sort by area, descending order
        areas = x_sizes * y_sizes
        _, indices = torch.sort(areas, descending=True)
        x_sizes = x_sizes[indices]
        y_sizes = y_sizes[indices]

        # place samples individually
        placement = Placement()
        density = 0
        for (x_size, y_size) in zip(x_sizes, y_sizes):
            x_size = float(x_size)
            y_size = float(y_size)
            dist_params = {"low": torch.tensor([-1.0, -1.0]), "high": torch.tensor([1.0-x_size, 1.0-y_size])}
            candidate_dist =  get_distribution("uniform", dist_params)
            
            for attempt_num in range(self.max_attempts_per_instance):
                candidate_pos = candidate_dist.sample()
                # print(instance_idx, attempt_num, candidate_pos)
                if placement.check_legality(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size):
                    placement.commit_instance(candidate_pos[0].item(), candidate_pos[1].item(), x_size, y_size)
                    break
            density += (x_size * y_size)/4.0
            if density >= self.stop_density:
                break
        
        positions = placement.get_positions()
        sizes = placement.get_sizes()
        num_instances = positions.shape[0]

        # sample number of terminals
        num_terminals = get_distribution(**self.num_terminals_dist).sample((num_instances,)).int() # TODO condition dist on area of instance per Rent's
        max_num_terminals = torch.max(num_terminals)
        terminal_offsets = self.get_terminal_offsets(sizes[:,0], sizes[:,1], max_num_terminals, reference="bottom_left")

        # generate edges
        terminal_positions = positions.unsqueeze(dim=1) + terminal_offsets # (V, T, 2)
        terminal_distances = self.get_terminal_distances(terminal_positions)
        edge_exists = get_distribution(**self.edge_dist).sample(terminal_distances) # (V, T, V, T)
        is_source = get_distribution(**self.source_terminal_dist).sample((num_instances, max_num_terminals))

        # delete edges between same instance
        edge_exists = self.process_edge_matrix(edge_exists, is_source)

        # TODO convert to edge list and generate attributes
        import ipdb; ipdb.set_trace()
        return 1
    
    def get_terminal_offsets(self, x_sizes, y_sizes, max_num_terminals, reference="center"):
        # TODO check reference points for these
        # NOTE here we assume reference point (for computing offset) is center of instance
        # NOTE outputs use whatever units are being used for sizes
        # x_sizes: (num_instances)
        # y_sizes: (num_instances)
        # max_num_terminals: int
        half_perim = (x_sizes + y_sizes)
        
        terminal_locations = get_distribution("uniform", {"low": 0, "high": half_perim}).sample((max_num_terminals,)) # (max_term, num_instances)
        terminal_flip = get_distribution("bernoulli", {"probs": 0.5}).sample((max_num_terminals, x_sizes.shape[0])) # (max_term, num_instances)
        terminal_flip = (2 * terminal_flip) - 1

        x_sizes = x_sizes.unsqueeze(dim=0)
        y_sizes = y_sizes.unsqueeze(dim=0)
        terminal_offset_x = torch.clamp(terminal_locations, torch.zeros_like(x_sizes), x_sizes) - (x_sizes/2) 
        terminal_offset_y = torch.clamp(terminal_locations-x_sizes, torch.zeros_like(y_sizes), y_sizes) - (y_sizes/2)

        terminal_offset_x = terminal_flip * terminal_offset_x
        terminal_offset_y = terminal_flip * terminal_offset_y

        if reference == "bottom_left":
            terminal_offset_x += x_sizes/2
            terminal_offset_y += y_sizes/2

        terminal_offset = torch.stack((terminal_offset_x, terminal_offset_y), dim=-1).movedim(1, 0) # (num_instances, max_term, 2)
        return terminal_offset

    def get_terminal_distances(self, terminal_positions, norm_order=1):
        # given global terminal positions (V, T, 2)
        # compute and return pairwise L1 distances between terminals
        # optionally, can specify x for Lx distance using norm_order
        V, T, _ = terminal_positions.shape
        t_pos_1 = terminal_positions.view(V, T, 1, 1, 2)
        t_pos_2 = terminal_positions.view(1, 1, V, T, 2)
        delta_pos = t_pos_1 - t_pos_2 # (V, T, V, T, 2)
        distance = torch.norm(delta_pos, p=norm_order, dim=-1) # (V, T, V, T)
        return distance

    def process_edge_matrix(self, edge_exists, is_source):
        # edge_existence tensor (V, T, V, T)
        # is_source int tensor (V, T) (0=sink, 1=source)
        # process  as follows:
        # remove all edges ending in a source
        # remove all edges originating from a sink
        # remove all edges between same instance
        V, T, _, _ = edge_exists.shape
        assert is_source.shape == edge_exists.shape[:2]
        source_filter = is_source.view(V, T, 1, 1)
        sink_filter = 1-is_source.view(1, 1, V, T)
        self_edge_filter = (1-torch.eye(V)).view(V, 1, V, 1)
        
        edges = edge_exists * source_filter
        edges = edges * sink_filter
        edges = edges * self_edge_filter
        return edges

    def generate_edge_list(self, edge_exists, terminal_offsets):
        # TODO
        return 1

class Placement:
    def __init__(self, x = None, sizes = None, mask = None):
        # initializes empty placement, unless x, sizes, and mask are specified
        # x is predicted placements (V, 2)
        # attr is width height (V, 2)
        self.insts = []
        self.x = []
        self.y = []
        self.x_size = []
        self.y_size = []
        self.is_port = []
        if (x is not None) and (sizes is not None) and (mask is not None):
            for size, loc, is_ports in zip(sizes, x, mask):
                if not is_ports:
                    self.insts.append(
                        shapely.box(loc[0], loc[1], loc[0] + size[0], loc[1] + size[1])
                    )
            self.x = list(x[:,0])
            self.y = list(x[:,1])
            self.x_size = list(sizes[:,0])
            self.y_size = list(sizes[:,1])
            self.is_port = list(mask)

        self.chip = shapely.box(-1, -1, 1, 1)
        self.eps = 1e-8

    def check_legality(self, x_pos, y_pos, x_size, y_size, score=False):
        # checks legality of current placement (or optionally current placement with candidate)
        # x_pos, y_pos, x_size, y_size are floats
        # returns float with legality of placement (1 = bad, 0 = legal), or bool if score=False
        insts = self.insts + [shapely.box(x_pos, y_pos, x_pos + x_size, y_pos + y_size)]

        insts_area = sum([i.area for i in insts])
        insts_overlap = shapely.intersection(shapely.unary_union(insts), self.chip).area

        if score:
            return insts_overlap/insts_area
        else:
            return abs(insts_overlap - insts_area) < self.eps
    
    def commit_instance(self, x_pos, y_pos, x_size, y_size, is_port=False):
        # adds instance with specified params without checking if legal or not
        if not is_port:
            self.insts.append(shapely.box(x_pos, y_pos, x_pos + x_size, y_pos + y_size))
        self.x.append(x_pos)
        self.y.append(y_pos)
        self.x_size.append(x_size)
        self.y_size.append(y_size)
        self.is_port.append(is_port)

    def get_density(self):
        insts_area = sum([i.area for i in self.insts])
        density = insts_area/self.chip.area
        return density
    
    def get_positions(self):
        # returns tensor(V, 2) of x,y placements
        positions = torch.stack((torch.tensor(self.x), torch.tensor(self.y)), dim=-1)
        return positions

    def get_sizes(self):
        # returns tensor(V, 2) of x,y sizes
        sizes = torch.stack((torch.tensor(self.x_size), torch.tensor(self.y_size)), dim=-1)
        return sizes
    
def plot_placement(positions, sizes, name = "debug_placement"):
    picture = utils.visualize(positions, sizes)
    utils.debug_plot_img(np.moveaxis(picture,-1, 0), "debug_placement")