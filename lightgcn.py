import torch
from torch import nn
from torch_geometric.data import Data
from lightgcncov import LightGConv
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self,train,num_u,num_v,num_layers = 2,dim = 64,reg=1e-4
                 ,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(LightGCN,self).__init__()
        self.M = num_u
        self.N = num_v
        self.num_layers = num_layers
        self.device = device
        self.reg = reg
        self.embed_dim = dim



        edge_user = torch.tensor(train['userId'].values-1) 
        # Make the index of the movie start from num_u
        edge_item = torch.tensor(train['movieId'].values-1)+self.M 
        
        # Create the undirected graph for the positive graph and negative graph
        edge = torch.stack((torch.cat((edge_user,edge_item),0),torch.cat((edge_item,edge_user),0)),0)
        self.data=Data(edge_index=edge)
        
        self.E1 = nn.Parameter(torch.empty(self.M + self.N, dim))
        nn.init.xavier_normal_(self.E1.data)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(LightGConv())         
        
        
    def get_embedding(self):
        # Generate embeddings z_p
        B=[]
        B.append(self.E1)
        x = self.convs[0](self.E1,self.data.edge_index)

        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x,self.data.edge_index)

            B.append(x)
        Z = sum(B)/len(B) 
        return Z
            
        
    
    def forward(self,u,v,n,device):
        emb = self.get_embedding()
        u_ = emb[u].to(device)
        v_ = emb[v].to(device)
        n_ = emb[n].to(device)
        positivebatch = torch.mul(u_ , v_ )
        negativebatch = torch.mul(u_.view(len(u_),1,self.embed_dim),n_)  
        BPR_loss =  F.logsigmoid(((positivebatch.sum(dim=1).view(len(u_),1))) - negativebatch.sum(dim=2)).sum(dim=1) # weight
        reg_loss = u_.norm(dim=1).pow(2).sum() + v_.norm(dim=1).pow(2).sum() + n_.norm(dim=2).pow(2).sum() 
        return -torch.sum(BPR_loss) + self.reg * reg_loss
            
