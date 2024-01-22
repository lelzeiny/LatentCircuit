# TODO:
# make into a nice class interface
# refactor utils so that policies control how logging takes place
import torch
import numpy as np
import time

def open_loop(batch_size, model, x_in, cond, intermediate_every = 200):
    # x_in: (B, V, F)
    # samples: (B, V, F)
    samples, intermediates = model.reverse_samples(batch_size, x_in, cond, intermediate_every = intermediate_every)
    return samples, intermediates

def open_loop_multi(model, x_in, cond, num_attempts, score_fn):
    # x_in: (B, V, F)
    # samples: (B, V, F)
    # generate batch of samples, only return the best according to score_fn
    # assumes 0 score is lowest possible
    B, V, F = x_in.shape
    assert B == 1, "open-loop (multi) policy cannot run in batched mode"

    samples, _ = model.reverse_samples(num_attempts, x_in, cond)
    intermediates = []
    argmax = 0
    max_score = 0
    for i in range(samples.shape[0]):
        score = score_fn(samples[i])
        if score > max_score:
            argmax = i
            max_score = score
        intermediates.append(samples[i:i+1])
    return samples[argmax:argmax+1], intermediates

def iterative(model, x_in, cond, score_fn, num_iter = 4):
    # sort nodes by decreasing size NOTE: experiment with other options for sorting?
    # each reverse sample produces candidate policy
    # going down list of nodes, find first node that produces legality conflict
    # then commit all nodes that do not produce conflict, masking them out
    # repeat reverse sampling for remaining nodes NOTE: experiment with backtracking?
    # finish once all nodes are masked, or iteration limit is reached
    # score_fn: (x, mask) -> bool True if legal
    # samples: (B, V, F)
    B, V, F = x_in.shape
    assert B == 1, "iterative policy cannot run in batched mode"

    # sort vertices by area
    instance_area = compute_instance_area(cond).cpu().numpy()
    instance_order = np.argsort(-instance_area) # index of largest instance first

    masks = [cond.is_ports]
    instances_committed = 0
    intermediates = []
    # import ipdb; ipdb.set_trace()
    gen_time = 0
    scoring_time = 0
    samples = x_in
    for iter_idx in range(num_iter):
        t0 = time.time()
        samples, _ = model.reverse_samples(1, samples, cond, mask_override = masks[-1])
        t1 = time.time()
        intermediates.append(samples)
        # update mask
        new_mask = masks[-1].clone()
        scoring_mask = masks[0]|(~masks[-1])
        # commit more instances until fail
        for i in range(instances_committed, V):
            instance = instance_order[i]
            scoring_mask[instance] = False # include next instance in scoring
            if masks[0][instance] or score_fn(samples[0], scoring_mask):
                # commit instance if it's a port or legal enough
                instances_committed += 1
                new_mask[instance] = True # don't move committed instances
            else:
                break
        masks.append(new_mask)
        t2 = time.time()
        gen_time += t1-t0
        scoring_time += t2-t1
        if torch.sum(new_mask) == V:
            break
    commit_rate = (torch.sum(masks[-1])-torch.sum(masks[0])) / (V-torch.sum(masks[0]))
    info = {"gen_t": gen_time, "scoring_t": scoring_time, "commit_rate": commit_rate, "iterations": iter_idx}
    return samples, intermediates, masks[1:], info

def compute_instance_area(cond):
    # cond.x: (V, F)
    # cond.is_ports: (V)
    # get 1D numpy array with sizes of each instance
    areas = cond.x[:,0] * cond.x[:,1]
    return areas