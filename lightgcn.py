
import torch
from torch import nn
from torch_geometric.data import Data
from lightgcncov import LightGConv
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self,train,num_u,num_v,threshold,num_layers = 2,MLP_layers=2,dim = 64,reg=1e-4
                 ,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(LightGCN,self).__init__()
        self.M = num_u
        self.N = num_v
        self.num_layers = num_layers
        self.MLP_layers = MLP_layers
        self.device = device
        self.reg = reg
        self.embed_dim = dim



        edge_user = torch.tensor(train[train['rating']>threshold]['userId'].values-1) 
        # Make the index of the movie start from num_u
        edge_item = torch.tensor(train[train['rating']>threshold]['movieId'].values-1)+self.M 
        edge_user_n = torch.tensor(train[train['rating']<=threshold]['userId'].values-1)
        edge_item_n = torch.tensor(train[train['rating']<=threshold]['movieId'].values-1)+self.M
        
        # Create the undirected graph for the positive graph and negative graph
        edge_p = torch.stack((torch.cat((edge_user,edge_item),0),torch.cat((edge_item,edge_user),0)),0)
        self.data_p=Data(edge_index=edge_p)
        edge_n = torch.stack((torch.cat((edge_user_n,edge_item_n),0),torch.cat((edge_item_n,edge_user_n),0)),0)
        self.data_n=Data(edge_index=edge_n)
        
        # For positive graph
        self.E1 = nn.Parameter(torch.empty(self.M + self.N, dim))
        nn.init.xavier_normal_(self.E1.data)
        
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(LightGConv()) 


        # For the negative graph
        self.E2 = nn.Parameter(torch.empty(self.M + self.N, dim))
        nn.init.xavier_normal_(self.E2.data)

        for _ in range(MLP_layers):
            self.mlps.append(nn.Linear(dim,dim,bias=True))
            nn.init.xavier_normal_(self.mlps[-1].weight.data)
        
        
        
        # Attntion model
        self.attn = nn.Linear(dim,dim,bias=True)
        self.q = nn.Linear(dim,1,bias=False)
        self.attn_softmax = nn.Softmax(dim=1)
        
        
        
        
    def get_embedding(self):
        # Generate embeddings z_p
        B=[]
        B.append(self.E1)
        x = self.convs[0](self.E1,self.data_p.edge_index)

        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x,self.data_p.edge_index)

            B.append(x)
        z_p = sum(B)/len(B) 

        # Generate embeddings z_n
        C = []
        C.append(self.E2)
        x = F.dropout(F.relu(self.mlps[0](self.E2)),p=0.5,training=self.training)
        for i in range(1,self.MLP_layers):
            x = self.mlps[i](x)
            x = F.relu(x)
            x = F.dropout(x,p=0.5,training=self.training)
            C.append(x)
        z_n = C[-1]
        
        # Attention for final embeddings Z
        w_p = self.q(F.dropout(torch.tanh((self.attn(z_p))),p=0.5,training=self.training))
        w_n = self.q(F.dropout(torch.tanh((self.attn(z_n))),p=0.5,training=self.training))
        alpha_ = self.attn_softmax(torch.cat([w_p,w_n],dim=1))
        Z = alpha_[:,0].view(len(z_p),1) * z_p + alpha_[:,1].view(len(z_p),1) * z_n
        return Z
            
        
    
    def forward(self,u,v,w,n,device):
        emb = self.get_embedding()
        u_ = emb[u].to(device)
        v_ = emb[v].to(device)
        n_ = emb[n].to(device)
        w_ = w.to(device)
        positivebatch = torch.mul(u_ , v_ )
        negativebatch = torch.mul(u_.view(len(u_),1,self.embed_dim),n_)  
        # (-1/2*torch.sign(w_)+3/2)).view(len(u_),1) = 2 if w_ > 3 else 0  a para set to punish the negative samples
        BPR_loss =  F.logsigmoid((((-1/2*torch.sign(w_)+3/2)).view(len(u_),1) * (positivebatch.sum(dim=1).view(len(u_),1))) - negativebatch.sum(dim=2)).sum(dim=1) # weight
        reg_loss = u_.norm(dim=1).pow(2).sum() + v_.norm(dim=1).pow(2).sum() + n_.norm(dim=2).pow(2).sum() 
        return -torch.sum(BPR_loss) + self.reg * reg_loss
            
