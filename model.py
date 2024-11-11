import torch
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree


class PositiveProp(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers=2):
        super(PositiveProp, self).__init__()
        self.lightgcn = LightGCN(num_nodes, embedding_dim, num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLu()
        )
        self.attention1 = nn.Linear(embedding_dim, embedding_dim)
        self.attention2 = nn.Linear(embedding_dim, 2)

    def forward(self, edge_index, edge_label_index):
        z_prime = self.lightgcn(edge_index, edge_label_index)
        z_double_prime = self.mlp(self.lightgcn.embedding.weight)

        attention_score_1 = F.tanh(self.attention1(z_prime))
        attention_score_1 = self.attention2(attention_score_1)
        attention_score_2 = F.tanh(self.attention1(z_double_prime))
        attention_score_2 = self.attention2(attention_score_2)

        attention_score = torch.cat([attention_score_1, attention_score_2], dim=1)
        attention_score = F.softmax(attention_score, dim=1)

        alpha_1 = attention_score[:, 0].unsqueeze(1).expand_as(self.lightgcn.embedding.weight)
        alpha_2 = attention_score[:, 1].unsqueeze(1).expand_as(self.lightgcn.embedding.weight)
        
        z = alpha_1 * z_prime + alpha_2 * z_double_prime
        return z
    
    
class NegativeProp(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers=2):
        super(NegativeProp, self).__init__()
        self.lightgcn = LightGCN(num_nodes, embedding_dim, num_layers)

    def forward(self, edge_index, edge_label_index):
        v = self.lightgcn(edge_index, edge_label_index)
        return v