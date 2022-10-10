import os
import re
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch import Tensor
from torch_geometric.data import Dataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull
from torch_geometric.io.planetoid import index_to_mask
from torch_geometric.transforms import NormalizeFeatures
from src.data.split import get_idx_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run at console -> python -c 'from src.data.data_utils import *; split_data("Cora", 5, 3, 85)'
def split_data(
        name: str, 
        samples_in_one_fold: int, 
        k_fold: int, 
        test_samples_per_class: int):
    """
    name: str, the name of the dataset
    samples_in_one_fold: int, sample x% of each class to one fold   
    k_fold: int, k-fold cross validation. One fold is used as validation the rest portions are used as training
    test_samples_per_class: int, sample x% of each class for test set
    """
    print(name)
    assert name in ['Cora','Citeseer', 'Pubmed', 'Computers', 'Photo', 'CS', 'Physics', 'CoraFull']
    if name in ['Cora','Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/', name=name, split='random')
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root='./data/', name=name)
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root='./data/', name=name)
    elif name == 'CoraFull':
        dataset = CoraFull(root='./data/')

    split_type = str(samples_in_one_fold)+"_"+str(k_fold)+'f_'+str(test_samples_per_class)       
    raw_dir = Path(os.path.join('data','split', str(name), split_type))
    raw_dir.mkdir(parents=True, exist_ok=True)

    # For each configuration we split the data five times
    for i in range(5):
        assert int(samples_in_one_fold)*int(k_fold)+int(test_samples_per_class) <= 100, "Invalid fraction" 
        k_fold_indices, test_indices = get_idx_split(dataset,
                    samples_per_class_in_one_fold=samples_in_one_fold/100.,
                    k_fold=k_fold,
                    test_samples_per_class=test_samples_per_class/100.)
        split_file = f'{name.lower()}_split_{i}.npz'
        print(f"sample/fold/test: {len(k_fold_indices[0])}/{len(k_fold_indices)}/{len(test_indices)}")
        np.savez(raw_dir/split_file, k_fold_indices=k_fold_indices, test_indices=test_indices)

def load_data(name: str, split_type: str, split: int, fold: int) -> Dataset:
    """
    name: str, the name of the dataset
    split_type: str, format {sample per fold ratio}_{k fold}_{test ratio}. For example, 5_3f_85
    split: int, index of the split. In total five splits were generated for each dataset. 
    fold: int, index of the fold to be used as validation set. The rest k-1 folds will be used as training set.
    """
    transform = NormalizeFeatures()
    if name in ['Cora','Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/', name=name, transform=transform)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(root='./data/', name=name, transform=transform)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name in ['CS', 'Physics']:
        dataset = Coauthor(root='./data/', name=name, transform=transform)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    elif name == 'CoraFull':
        dataset = CoraFull(root='./data/', transform=transform)
        load_split_from_numpy_files(dataset, name, split_type, split, fold)
    return dataset

def load_split_from_numpy_files(dataset, name, split_type, split, fold):
    """
    load train/val/test from saved k-fold split files
    """
    raw_dir = Path(os.path.join('data','split', str(name), split_type))
    assert raw_dir.is_dir(), "Split type does not exist."
    split_file = f'{name.lower()}_split_{split}.npz'
    masks = np.load(raw_dir / split_file, allow_pickle=True)
    val_indices = masks['k_fold_indices'][fold]
    train_indices = np.concatenate(np.delete(masks['k_fold_indices'], fold, axis=0))
    test_indices = masks['test_indices']
    dataset.data.train_mask = index_to_mask(train_indices, dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(val_indices, dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(test_indices, dataset.data.num_nodes)

# Run at console -> python -c 'from src.data.data_utils import *; generate_node_to_nearest_training("Cora", "5_3f_85")'
def generate_node_to_nearest_training(name: str, split_type: str, bfs_depth = 10):
    max_split = int(split_type.split("_")[0])
    max_fold = int(split_type.split("_")[1].replace("f",""))
    for split in tqdm(range(max_split)):
        raw_dir = Path(os.path.join('data','dist_to_train', str(name), split_type))
        for fold in tqdm(range(max_fold)):
            dataset = load_data(name=name, split_type=split_type, split=split, fold=fold)
            data = dataset.data
            dist_to_train = torch.ones(data.num_nodes) * bfs_depth
            dist_to_train = shortest_path_length(data.edge_index, data.train_mask, bfs_depth)
            raw_split_dir = raw_dir / f'split_{split}'
            raw_split_dir.mkdir(parents=True, exist_ok=True)
            split_file = f'{name.lower()}_dist_to_train_f{fold}.npy'
            np.save(raw_split_dir/split_file, dist_to_train)

def load_node_to_nearest_training(name: str, split_type: str, split: int, fold: int):
    split_file = os.path.join(
        'data', 'dist_to_train', str(name), split_type, f'split_{split}',
        f'{name.lower()}_dist_to_train_f{fold}.npy')
    if not os.path.isfile(split_file):
        generate_node_to_nearest_training(name, split_type)
    return torch.from_numpy(np.load(split_file))

def shortest_path_length(edge_index, mask, max_hop):
    """
    Return the shortest path length to the mask for every node
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=mask.device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
        for node in current_hop:
            node_mask = edge_index[0,:]==node
            nbrs = edge_index[1,node_mask]
            next_hop[nbrs] = True
        hop += 1
        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train        

def get_train_hop_hist(
        edge_index: np.ndarray, train_index: np.ndarray, nodes: int,
        max_hop: int
) -> np.ndarray:
    train_hop_count = np.zeros([nodes, max_hop + 1], dtype=np.int32)
    for t in train_index:
        hops = np.full(nodes, fill_value=max_hop, dtype=np.int32)
        current_nodes = {t}
        seen_nodes = set()
        for h in range(max_hop):
            if not current_nodes:
                break
            current_idx = np.asarray(list(current_nodes))
            hops[current_idx] = h
            seen_nodes |= current_nodes
            next_nodes = set()
            for n in current_nodes:
                next_nodes |= set(
                    edge_index[1, edge_index[0, :] == n].tolist()
                ) - seen_nodes
            current_nodes = next_nodes
        train_hop_count[np.arange(nodes), hops] += 1
    return train_hop_count

def load_train_hop_hist(
        name: str, split_type: str, split: int, fold: int, max_hop: int
) -> Tensor:
    dataset = load_data(name, split_type, split, fold)
    cache_dir = os.path.join(
        'data', 'train_hop_dist', str(name), split_type)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_name = os.path.join(cache_dir, f's{split}_f{fold}_h{max_hop}.npy')
    if os.path.isfile(cache_name):
        print(f'loading train_hop_dist from {cache_name}')
        return torch.from_numpy(np.load(cache_name)).to(torch.get_default_dtype())
    else:
        print(f'computing train_hop_dist ...')
        data = dataset.data
        nodes = data.num_nodes
        train_index = np.arange(nodes)[data.train_mask.cpu().numpy()]
        train_hop_dist = get_train_hop_hist(
            data.edge_index.cpu().numpy(), train_index, nodes, max_hop)
        print(f'saving computed train_hop_dist to {cache_name}')
        np.save(cache_name, train_hop_dist)
        return torch.from_numpy(train_hop_dist).to(torch.get_default_dtype())
