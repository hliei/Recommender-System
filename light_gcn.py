import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split


class LightGCNModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim,initial_embedding,num_layers=5):
        super(LightGCNModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.lightgcn = LightGCN(num_nodes=num_users+num_items,embedding_dim=embedding_dim, num_layers=num_layers)
        with torch.no_grad():
            self.lightgcn.embedding.weight = torch.nn.Parameter(initial_embedding.to(torch.float64))

    def forward(self, edge_index):
        # Combine user and item embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        # Pass through LightGCN layer
        #x=int(x)
        #x=torch.LongTensor(x)
        edge_index=edge_index.to(torch.int64)
        x = self.lightgcn(edge_index)
        return x

    def predict(self, user_index, item_index):
        user_embedding = self.user_embedding(user_index)
        item_embedding = self.item_embedding(item_index)
        return (user_embedding * item_embedding).sum(dim=1)