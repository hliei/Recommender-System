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
            nn.ReLU()
        )
        self.attention1 = nn.Linear(embedding_dim, embedding_dim)
        self.attention2 = nn.Linear(embedding_dim, 2)

    def forward(self, edge_index, edge_label_index):
        z_prime = self.lightgcn(edge_index, edge_label_index)
        z_prime = self.lightgcn.get_embedding(edge_index)
        z_double_prime = self.mlp(self.lightgcn.embedding.weight)

        attention_score_1 = F.tanh(self.attention1(z_prime))
        attention_score_1 = self.attention2(attention_score_1)
        attention_score_2 = F.tanh(self.attention1(z_double_prime))
        attention_score_2 = self.attention2(attention_score_2)

        attention_score = torch.cat([attention_score_1, attention_score_2], dim=1)
        attention_score = F.softmax(attention_score, dim=1)

        alpha_1 = attention_score[:, 0].unsqueeze(1).expand_as(z_prime)
        alpha_2 = attention_score[:, 1].unsqueeze(1).expand_as(z_double_prime)
        
        z_emb = alpha_1 * z_prime + alpha_2 * z_double_prime

        out_src = z_emb[edge_label_index[0]]
        out_dst = z_emb[edge_label_index[1]]
        z = (out_src * out_dst).sum(dim=-1)
        return z
    
    def loss_fn(self, z, edge_label_index):
        z_i, z_j = z.chunk(2)
        return self.lightgcn.recommendation_loss(z_i, z_j, node_id=edge_label_index.unique())
    
    def get_embedding(self,edge_index):
        z_prime = self.lightgcn.get_embedding(edge_index)
        z_double_prime = self.mlp(self.lightgcn.embedding.weight)

        attention_score_1 = F.tanh(self.attention1(z_prime))
        attention_score_1 = self.attention2(attention_score_1)
        attention_score_2 = F.tanh(self.attention1(z_double_prime))
        attention_score_2 = self.attention2(attention_score_2)

        attention_score = torch.cat([attention_score_1, attention_score_2], dim=1)
        attention_score = F.softmax(attention_score, dim=1)

        alpha_1 = attention_score[:, 0].unsqueeze(1).expand_as(z_prime)
        alpha_2 = attention_score[:, 1].unsqueeze(1).expand_as(z_double_prime)
        
        z_emb = alpha_1 * z_prime + alpha_2 * z_double_prime

        return z_emb
    
    
class NegativeProp(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_layers=2):
        super(NegativeProp, self).__init__()
        self.lightgcn = LightGCN(num_nodes, embedding_dim, num_layers)

    def forward(self, edge_index, edge_label_index):
        v = self.lightgcn(edge_index, edge_label_index)
        return v
    
    def loss_fn(self, z, edge_label_index):
        z_i, z_j = z.chunk(2)
        return self.lightgcn.recommendation_loss(z_i, z_j,node_id = edge_label_index.unique())
    


class FinalModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers=2):
        super(FinalModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.positive_prop = PositiveProp(self.num_nodes, embedding_dim, num_layers)
        self.negative_prop = NegativeProp(self.num_nodes, embedding_dim, num_layers)

    def forward(self, pos_edge_index, neg_edge_index, pos_edge_label_index, neg_edge_label_index):

        z = self.positive_prop(pos_edge_index, pos_edge_label_index)
        v = self.negative_prop(neg_edge_index, neg_edge_label_index)

        return z, v 
    
    def bpr_loss(self,z,v,pos_edge_label_index,neg_edge_label_index):
        return self.positive_prop.loss_fn(z,pos_edge_label_index), self.negative_prop.loss_fn(v,neg_edge_label_index)
    
    def get_2_embedding(self,pos_edge_index, neg_edge_index):
        return self.positive_prop.get_embedding(pos_edge_index),self.negative_prop.lightgcn.get_embedding(neg_edge_index)