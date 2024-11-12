import torch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from torch_geometric.datasets import MovieLens100K
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree

from model import FinalModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_movielens_100k(path):
    df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df['user_id'] -= 1  
    df['item_id'] -= 1  
    return df

def create_edge_data(df):
    user_item_interactions = df[['user_id', 'item_id']].to_numpy().T
    user_item_rating = df['rating'].astype(float).to_numpy()
    edge_index = torch.tensor(user_item_interactions, dtype=torch.long) 
    edge_weight = torch.tensor(user_item_rating, dtype=torch.float32)
    pos_edge_index, neg_edge_index = get_2_edge_index(edge_index,edge_weight)
    return edge_index, edge_weight, pos_edge_index, neg_edge_index

def get_2_edge_index(edge_index,edge_weight):
    pos_index = edge_weight>=3
    neg_index = edge_weight<3
    pos_edge_index = edge_index[:,pos_index]
    neg_edge_index = edge_index[:,neg_index]
    return pos_edge_index, neg_edge_index

# Load the dataset
movielens_path = 'dataset/raw/u.data'
dataset = load_movielens_100k(movielens_path)
num_users, num_items = dataset['user_id'].nunique(), dataset['item_id'].nunique()
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_edge_index, train_edge_weight, train_pos_edge_index, train_neg_edge_index = create_edge_data(train_data)
train_edge_index = train_edge_index.to(device)

batch_size = 256
train_loader = torch.utils.data.DataLoader(
    range(train_edge_index.size(1)),
    shuffle=True,
    batch_size=batch_size,
)

model = FinalModel(num_users , num_items, embedding_dim=32, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        batch_edge_index = train_edge_index[:,index]
        batch_edge_weight = train_edge_weight[index]
        pos_index = batch_edge_weight>=3
        neg_index = batch_edge_weight<3
        pos_edge_index = batch_edge_index[:,pos_index]
        neg_edge_index = batch_edge_index[:,neg_index]
        pos_edge_index_j = torch.stack([
            pos_edge_index[0],
            torch.randint(num_users, num_users + num_items,
                          (len(pos_edge_index[0]), ), device=device)
        ], dim=0)
        neg_edge_index_j = torch.stack([
            neg_edge_index[0],
            torch.randint(num_users, num_users + num_items,
                          (len(neg_edge_index[0]), ), device=device)
        ], dim=0)
        pos_edge_label_index = torch.cat([
            pos_edge_index,
            pos_edge_index_j,
        ], dim=1)
        neg_edge_label_index = torch.cat([
            neg_edge_index,
            neg_edge_index_j,
        ], dim=1)
        optimizer.zero_grad()
        z,v = model(train_pos_edge_index, train_neg_edge_index,pos_edge_label_index, neg_edge_label_index)
        z_i, z_j = z.chunk(2)
        v_i, v_j = v.chunk(2)
        loss1,loss2 = model.bpr_loss(z,v,pos_edge_label_index, neg_edge_label_index)
        loss = loss1+loss2
        loss.backward()
        optimizer.step()
        total_loss += float(loss1) * z_i.numel() + float(loss2) * v_i.numel()
        total_examples += z_i.numel() + v_i.numel()
    return total_loss / total_examples

@torch.no_grad()
def evaluate(edge_index,pos_edge_index,neg_edge_index,num_users,K=5, thershold=0.5):
    pos_emb, neg_emb = model.get_2_embedding(pos_edge_index, neg_edge_index)
    pos_user_emb, pos_item_emb = pos_emb[:num_users], pos_emb[num_users:]
    neg_user_emb, neg_item_emb = neg_emb[:num_users], neg_emb[num_users:]

    precision = recall = total_examples = 0
    for start in range(0, num_users, batch_size):
        end = start + batch_size
        logits_pos = pos_user_emb[start:end] @ pos_item_emb.t()
        logits_neg = neg_user_emb[start:end] @ neg_item_emb.t()
        logits_pos = torch.sigmoid(logits_pos)
        logits_neg = torch.sigmoid(logits_neg)

        # Exclude examples from training set
        mask_pos = ((pos_edge_index[0] >= start) &
                (pos_edge_index[0] < end))
        logits_pos[pos_edge_index[0, mask_pos] - start,
               pos_edge_index[1, mask_pos] - num_users] = float('-inf')
        mask_neg = ((neg_edge_index[0] >= start) &
                (neg_edge_index[0] < end))
        logits_neg[neg_edge_index[0, mask_neg] - start,
                neg_edge_index[1, mask_neg] - num_users] = float('-inf')
        
        # Filter out user-item set for logits_neg < thershold
        mask = logits_neg < thershold 
        logits_pos_filtered = logits_pos * mask

        # Ground truth
        ground_truth = torch.zeros_like(logits_pos_filtered, dtype=torch.bool)
        mask = ((edge_index[0] >= start) &
                (edge_index[0] < end))
        ground_truth[edge_index[0, mask] - start,
                     edge_index[1, mask] - num_users] = True
        node_count = degree(edge_index[0, mask] - start,
                            num_nodes=logits_pos_filtered.size(0))
        
        # Calculate metrics
        topk_index = logits_pos_filtered.topk(K, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)
        precision += float((isin_mat.sum(dim=-1) / K).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples

for epoch in range(1, 3):
    loss = train()
    precision, recall = evaluate(train_edge_index, train_pos_edge_index, train_neg_edge_index, num_users, K=15)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@K: '
          f'{precision:.4f}, Recall@K: {recall:.4f}')

# Evaluate on the test set
test_edge_index, test_edge_weight, test_pos_edge_index, test_neg_edge_index = create_edge_data(test_data)
test_edge_index = test_edge_index.to(device)
test_pos_edge_index = test_pos_edge_index.to(device)
test_neg_edge_index = test_neg_edge_index.to(device)
model.eval()
@torch.no_grad()
def evaluate_test(edge_index,pos_edge_index,neg_edge_index,K=5, thershold=0.5):
    pos_emb, neg_emb = model.get_2_embedding(pos_edge_index, neg_edge_index)
    pos_user_emb, pos_item_emb = pos_emb[:num_users], pos_emb[num_users:]
    neg_user_emb, neg_item_emb = neg_emb[:num_users], neg_emb[num_users:]
    
    precision = recall = total_examples = 0
    for start in range(0, num_users, batch_size):
        end = start + batch_size
        logits_pos = pos_user_emb[start:end] @ pos_item_emb.t()
        logits_neg = neg_user_emb[start:end] @ neg_item_emb.t()
        logits_pos = torch.sigmoid(logits_pos)
        logits_neg = torch.sigmoid(logits_neg)

        # Exclude examples from training set
        mask_pos = ((pos_edge_index[0] >= start) &
                (pos_edge_index[0] < end))
        logits_pos[pos_edge_index[0, mask_pos] - start,
               pos_edge_index[1, mask_pos] - num_users] = float('-inf')
        mask_neg = ((neg_edge_index[0] >= start) &
                (neg_edge_index[0] < end))
        logits_neg[neg_edge_index[0, mask_neg] - start,
                neg_edge_index[1, mask_neg] - num_users] = float('-inf')
        
        # Filter out user-item set for logits_neg < thershold
        mask = logits_neg < thershold 
        logits_pos_filtered = logits_pos * mask

        # Ground truth
        ground_truth = torch.zeros_like(logits_pos_filtered, dtype=torch.bool)
        mask = ((edge_index[0] >= start) &
                (edge_index[0] < end))
        ground_truth[edge_index[0, mask] - start,
                     edge_index[1, mask] - num_users] = True
        node_count = degree(edge_index[0, mask] - start,
                            num_nodes=logits_pos_filtered.size(0))
        
        # Calculate metrics
        topk_index = logits_pos_filtered.topk(K, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)
        precision += float((isin_mat.sum(dim=-1) / K).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples

precision, recall = evaluate_test(test_edge_index, test_pos_edge_index, test_neg_edge_index, K=15)
print(f'Test Precision@K: {precision:.4f}, Test Recall@K: {recall:.4f}')

