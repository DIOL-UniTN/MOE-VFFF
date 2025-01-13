import torch
from tqdm import tqdm
import numpy as np
from typing import List
import torch.nn.functional as F

def greedy_virtualization(search_type:str, leaf_weights:torch.Tensor, fim:torch.Tensor, n_pagesamples:int, 
                          n_virtualpages:int, n_splits:int, verbose:bool=True):
    if search_type == 'greedy_randomleafsplit':
        return greedy_randomleafsplit_match_virtualpage(leaf_weights, fim, n_splits, n_pagesamples, n_virtualpages, verbose)
    elif search_type == 'greedy_random':
        return greedy_random_match_virtualpage(leaf_weights, fim, n_pagesamples, n_virtualpages)
    elif search_type == 'greedy_fisher':
        return greedy_fisher_match_virtualpage(leaf_weights, fim, n_pagesamples, n_virtualpages)
    elif search_type == 'greedy_fisherleaf':
        return greedyefisherleaf_match_virtualpage(leaf_weights, fim, n_pagesamples, n_virtualpages)
    elif search_type == 'greedy_fisherleafinit':
        return greedy_fisherleafinit_match_virtualpage(leaf_weights, fim, n_pagesamples, n_virtualpages)

@torch.no_grad()
def merge_weights(weights, virtualpage_matches:List[List[int]], weights_per_page, n_splits:int) -> torch.Tensor:
    original_leaf_shape = weights.shape
    weightpages, pad_amount = pad_weights(weights, n_splits, int(weights_per_page))
    weightpages = weightpages.reshape(-1, int(weights_per_page))
    for matches in virtualpage_matches:
        weightpages[[matches]] = weightpages[[matches]].mean(dim=0)
    weightpages = weightpages.reshape(n_splits, -1)[:, :-pad_amount].flatten() if pad_amount else weightpages
    return weightpages.reshape(original_leaf_shape)

def pad_weights(weights:torch.Tensor, n_splits:int, weights_per_page:int) -> torch.Tensor:
    weightpages = weights.flatten().reshape(n_splits, -1)
    pad_amount = (weights_per_page - weightpages.size(1) % weights_per_page) if weightpages.size(1) % weights_per_page else 0
    return F.pad(weightpages, (0, int(pad_amount))).squeeze(), int(pad_amount)

def greedy_expand_match_virtualpage(leaf_weights:torch.Tensor, n_pagesamples:int, n_virtualpages:int, verbose:bool=True):
    import itertools
    logging.info("Calculating the cost of all weight-page combinations...")
    weightpages = leaf_weights.reshape(n_pagesamples, -1)
    weightpage_combinations = list(itertools.combinations(range(n_pagesamples), 2))
    weightpage_costs = []
    for i in tqdm(range(n_pagesamples), disable=(not verbose)):
        weightpage_costs.append((weightpages[i]-weightpages[i+1:]).abs().sum(dim=1).flatten())
    weightpage_costs = torch.cat(weightpage_costs).tolist()
    assert len(weightpage_costs) == len(weightpage_combinations)
    weightpage_costs_combinations_sorted = sorted(zip(weightpage_costs, weightpage_combinations))

    logging.info("Matching...")
    min_vp_cost, vp_idx = 0.0, 0
    virtualpage_matches, virtualpage_costs = [], []
    for cost, wp_indices in tqdm(weightpage_costs_combinations_sorted):
        both_matched_different = is_both_matched_different(wp_indices, virtualpage_matches)
        if both_matched_different:
            continue
        matched_vp_idx = match_ifone_already_matched(wp_indices, virtualpage_matches)
        if matched_vp_idx != None:
            virtualpage_cost = increase_virtualpage_cost_bymatch(matched_vp_idx, virtualpage_costs, cost)
        else:
            virtualpage_matches.append([wp_indices[0], wp_indices[1]])
            virtualpage_costs.append((cost, n_virtualpages))
            n_virtualpages += 1
        virtualpage_costs.sort()
        min_vp_cost, vp_idx = virtualpage_costs[0]
    total_cost = sum([cost for cost, _ in virtualpage_costs])
    return total_cost, virtualpage_matches

def greedy_base_match_virtualpage(leaf_weights:torch.Tensor, fim:torch.Tensor, n_pagesamples:int, n_virtualpages:int):
    weightpages = leaf_weights.reshape(n_pagesamples, -1)
    _, mag_wp_indices = weightpages.abs().sum(dim=1).sort()
    all_virtualpage_matches = [[i] for i in mag_wp_indices[:n_virtualpages]] 
    fimpages = fim.reshape(n_pagesamples, -1)

    total_cost = 0
    for i in tqdm(mag_wp_indices[n_virtualpages+1:]): # Match the remaining
        min_matching_cost = 1e10
        matched_vp_idx, matching_cost = find_best_virtualpage(i, weightpages, fimpages,
                                                                   all_virtualpage_matches)
        all_virtualpage_matches[matched_vp_idx].append(i)
        total_cost += matching_cost
    return total_cost, all_virtualpage_matches

def greedy_fisher_match_virtualpage(leaf_weights:torch.Tensor, fim:torch.Tensor, n_pagesamples:int, n_virtualpages:int):
    weightpages = leaf_weights.reshape(n_pagesamples, -1)
    fimpages = fim.reshape(n_pagesamples, -1)
    _, fim_wp_indices = fimpages.abs().sum(dim=1).sort()
    all_virtualpage_matches = [[i] for i in fim_wp_indices[:n_virtualpages].tolist()]
    fim_wp_indices = fim_wp_indices[n_virtualpages:].tolist()
    np.random.shuffle(fim_wp_indices)

    total_cost = 0
    for i in tqdm(fim_wp_indices): # NOTE: Vary order: rnadom, fisher, inorder
        matched_vp_idx, matching_cost = find_best_virtualpage(i, weightpages, fimpages,
                                                                   all_virtualpage_matches)
        all_virtualpage_matches[matched_vp_idx].append(i)
        total_cost += matching_cost
    return total_cost, all_virtualpage_matches

def greedy_fisherleaf_match_virtualpage(leaf_weights:torch.Tensor, fim:torch.Tensor, n_pagesamples:int, n_virtualpages:int, n_leaves:int):
    weightpages = leaf_weights.reshape(n_pagesamples, -1)
    n_pagesamples_eachleaf = n_pagesamples//n_leaves
    fimpages = fim.reshape(n_leaves, n_pagesamples_eachleaf, -1)
    fimpages_leaves_sorted, fimpages_leaves_indices = fimpages.abs().sum(dim=(1,2)).sort()

    total_cost, n_matches = 0.0, 0
    all_virtualpage_matches = []
    for i in tqdm(fimpages_leaves_indices):
        _, fimpage_leaf_indices = fimpages[i].abs().sum(dim=1).sort()
        fimpage_leaf_indices = fimpage_leaf_indices.tolist()
        np.random.shuffle(fimpage_leaf_indices)
        for j in tqdm(fimpage_leaf_indices, leave=False):
            j += n_pagesamples_eachleaf*i
            if len(all_virtualpage_matches) < n_virtualpages:
                all_virtualpage_matches.append([j])
                n_matches += 1
                continue
            matched_vp_idx, matching_cost = (j, weightpages, 
                                                                       fimpages.view(n_pagesamples, -1),
                                                                       all_virtualpage_matches)
            all_virtualpage_matches[matched_vp_idx].append(j)
            total_cost += matching_cost
            n_matches += 1
    return total_cost, all_virtualpage_matches

def greedy_fisherleafinit_match_virtualpage(leaf_weights:torch.Tensor, fim:torch.Tensor, n_pagesamples:int, n_virtualpages:int, n_leaves:int):
    weightpages = leaf_weights.reshape(n_pagesamples, -1)
    n_pagesamples_eachleaf = n_pagesamples//n_leaves

    fimpages = fim.reshape(n_leaves, n_pagesamples_eachleaf, -1)
    fimpages_leaves_sorted, fimpages_leaves_indices = fimpages.abs().sum(dim=2).sort()
    fimpages_leaves_notsorted_indices = torch.arange(fimpages_leaves_sorted.numel()).reshape(fimpages_leaves_sorted.shape)

    all_virtualpage_matches = []
    n_leaf_each_vp = (n_virtualpages + n_leaves - 1)//n_leaves

    # Initialize vps
    fimpages = fim.reshape(n_pagesamples, -1)
    total_cost, n_matches, init_vps = 0.0, 0, []
    for i in range(n_leaves):
        for j in range(n_leaf_each_vp):
            cur_idx = fimpages_leaves_indices[i, j] + i*n_pagesamples_eachleaf
            all_virtualpage_matches.append([cur_idx.item()])
            init_vps.append(cur_idx.item())
            n_matches += 1
            if n_matches == n_virtualpages:
                break
        if n_matches == n_virtualpages:
            break

    random_wp_indices = list(range(n_pagesamples))
    np.random.shuffle(random_wp_indices)
    random_wp_indices = [i for i in random_wp_indices if i not in init_vps]
    for i in tqdm(random_wp_indices): # NOTE: Vary order: rnadom, fisher, inorder
        matched_vp_idx, matching_cost = find_best_virtualpage(i, weightpages, fimpages,
                                                                   all_virtualpage_matches)
        all_virtualpage_matches[matched_vp_idx].append(i)
        total_cost += matching_cost
    return total_cost, all_virtualpage_matches

def greedy_random_match_virtualpage(leaf_weights:torch.Tensor, fim:torch.Tensor, n_pagesamples:int, n_virtualpages:int):
    weightpages = leaf_weights.reshape(n_pagesamples, -1)
    random_wp_indices = list(range(n_pagesamples))
    np.random.shuffle(random_wp_indices)
    all_virtualpage_matches = [[i] for i in random_wp_indices[:n_virtualpages]] # Match random n-th for initializing
    fimpages = fim.reshape(n_pagesamples, -1)

    total_cost = 0
    for i in tqdm(random_wp_indices[n_virtualpages:]): # Match the remaining
        matched_vp_idx, matching_cost = find_best_virtualpage(i, weightpages, fimpages,
                                                                   all_virtualpage_matches)
        all_virtualpage_matches[matched_vp_idx].append(i)
        total_cost += matching_cost
    return total_cost, all_virtualpage_matches

def greedy_randomleafsplit_match_virtualpage(leaf_weights:torch.Tensor, fim:torch.Tensor, n_splits:int, n_pagesamples:int, 
                                             n_virtualpages:int, verbose:bool=True):
    n_pagesamples_persplit = int(n_pagesamples//n_splits)
    weightpages = leaf_weights.reshape(n_splits, n_pagesamples_persplit, -1)
    fimpages = fim.reshape(n_splits, n_pagesamples_persplit, -1)
    n_virtualpages_persplit = n_virtualpages//n_splits
    vp_sizes = (n_virtualpages*fimpages.sum(dim=(1,2))/fimpages.sum()).int().tolist()

    # Initializing virtualpages
    # NOTE: Does giving half makes sense. Can portion according to fisher as well.
    random_wp_indices_splits = []
    all_virtualpage_matches_splits = []
    for split_idx in range(n_splits):
        random_wp_indices = list(range(n_pagesamples_persplit))
        np.random.shuffle(random_wp_indices)
        all_virtualpage_matches_tmp = [[i] for i in random_wp_indices[:vp_sizes[split_idx]]] # Match random n-th for initializing
        all_virtualpage_matches_splits.append(all_virtualpage_matches_tmp)
        random_wp_indices_splits.append(random_wp_indices)

    total_cost = 0
    all_virtualpage_matches = []
    for split_idx in tqdm(range(n_splits), disable=(not verbose)):
        random_wp_indices = random_wp_indices_splits[split_idx]
        verbose = True
        for i in tqdm(random_wp_indices[vp_sizes[split_idx]:], leave=False, disable=(not verbose)): # Match the remaining
            matched_vp_idx, matching_cost = find_best_virtualpage(i, weightpages[split_idx], fimpages[split_idx],
                                                                       all_virtualpage_matches_splits[split_idx])
            all_virtualpage_matches_splits[split_idx][matched_vp_idx].append(i)
            total_cost += matching_cost

        # Postprocess for index fix and for merging them back in one split
        for each_match in all_virtualpage_matches_splits[split_idx]:
            each_match = [i+n_pagesamples_persplit*split_idx 
                          for i in each_match] # We fix the leaf indices to flattened version since our model works that way
            all_virtualpage_matches.append(each_match)
    return total_cost, all_virtualpage_matches


def not_multi(args):
    wp_idx, weightpages, fimpages, all_virtualpage_matches = args
    min_matching_cost = 1e10
    for vp_idx, virtualpage_matches in enumerate(all_virtualpage_matches):
        matching_cost = ((weightpages[wp_idx]-weightpages[[virtualpage_matches]]).square()
                         *(fimpages[wp_idx]+fimpages[[virtualpage_matches]])).sum() # NOTE: do smt else than sume?
        if matching_cost < min_matching_cost:
            min_matching_cost = matching_cost
            matched_vp_idx = vp_idx
    return matched_vp_idx, min_matching_cost

def find_best_virtualpage_single(wp_idx:int, weightpages:torch.Tensor, 
                          fimpages:torch.Tensor,
                          all_virtualpage_matches:List[List[int]]):
    len_vp = len(all_virtualpage_matches)
    min_matching_cost = 1e10
    for vp_idx, virtualpage_matches in enumerate(all_virtualpage_matches):
        matching_cost = ((weightpages[wp_idx]-weightpages[[virtualpage_matches]]).square()
                         *(fimpages[wp_idx]+fimpages[[virtualpage_matches]])).sum() # NOTE: do smt else than sume?
        if matching_cost < min_matching_cost:
            min_matching_cost = matching_cost
            matched_vp_idx = vp_idx
    return matched_vp_idx, min_matching_cost

def find_best_virtualpage(wp_idx:int, weightpages:torch.Tensor, 
                          fimpages:torch.Tensor,
                          all_virtualpage_matches:List[List[int]]):

    # Flatten to parallalize
    flattened_all_vp_matches = []
    for matches in all_virtualpage_matches:
        flattened_all_vp_matches += matches

    vp_len_list = [len(matches) for matches in all_virtualpage_matches] # For decoding the costs
    matching_costs = ((weightpages[wp_idx]-weightpages[flattened_all_vp_matches]).square()
                     *(fimpages[wp_idx]+fimpages[flattened_all_vp_matches])).sum(dim=1).tolist()

    cur_idx, costs = 0, []
    for len_vp in vp_len_list:
        costs.append(sum(matching_costs[cur_idx:cur_idx+len_vp]))
        cur_idx += len_vp

    min_matching_cost, matched_vp_idx = torch.min(torch.tensor(costs), dim=0)
    return matched_vp_idx.item(), min_matching_cost.item()

def calculate_matching_cost(wp_idx:int, weightpages:torch.Tensor, 
                          fimpages:torch.Tensor,
                          all_virtualpage_matches:List[List[int]]):

    # Flatten to parallalize
    flattened_all_vp_matches = []
    for matches in all_virtualpage_matches:
        flattened_all_vp_matches += matches
    vp_len_list = [len(matches) for matches in all_virtualpage_matches] # For decoding the costs
    matching_costs = ((weightpages[wp_idx]-weightpages[flattened_all_vp_matches]).square()
                     *(fimpages[wp_idx]+fimpages[flattened_all_vp_matches])).sum(dim=1)

    cur_idx, costs = 0, []
    for len_vp in vp_len_list:
        costs.append(matching_costs[cur_idx:cur_idx+len_vp].sum())
        cur_idx += len_vp

    min_matching_cost, matched_vp_idx = torch.min(torch.tensor(costs), dim=0)
    return matched_vp_idx.item(), min_matching_cost.item()

def increase_virtualpage_cost_bymatch(vp_idx, virtualpage_costs, cost):
    for loc, (_, cur_vp_idx) in enumerate(virtualpage_costs):
        if cur_vp_idx == vp_idx:
            vp_loc = loc
    matched_vp_cost, matched_vp_idx = virtualpage_costs[vp_loc]
    virtualpage_costs[vp_loc] = (matched_vp_cost+cost, matched_vp_idx)
    return virtualpage_costs

def is_both_matched_different(wp_indices, all_virtualpage_matches):
    for vp_idx, cur_virtualpage_matches in enumerate(all_virtualpage_matches):
        first_matched, second_matched = wp_indices[0] in cur_virtualpage_matches, wp_indices[1] in cur_virtualpage_matches
        if first_matched and second_matched:
            return False
        elif first_matched and is_matched(wp_indices[1], all_virtualpage_matches):
            return True
        elif second_matched and is_matched(wp_indices[0], all_virtualpage_matches):
            return True
    return False

def is_matched(wp_id:int, all_virtualpage_matches):
    for vp_idx, cur_virtualpage_matches in enumerate(all_virtualpage_matches):
        matched = wp_id in cur_virtualpage_matches
        if matched:
            return True
    return False

def match_ifone_already_matched(wp_indices, all_virtualpage_matches):
    for vp_idx, cur_virtualpage_matches in enumerate(all_virtualpage_matches):
        first_matched, second_matched = wp_indices[0] in cur_virtualpage_matches, wp_indices[1] in cur_virtualpage_matches
        if first_matched and second_matched:
            return vp_idx
        elif first_matched:
            cur_virtualpage_matches.append(wp_indices[1])
            return vp_idx
        elif second_matched:
            cur_virtualpage_matches.append(wp_indices[0])
            return vp_idx
    return None
