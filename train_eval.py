import os.path as osp

import torch
from tqdm import tqdm

from torch_geometric.datasets import MovieLens100K
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the dataset
dataset = MovieLens100K(root='data/MovieLens100K')
data = dataset[0]