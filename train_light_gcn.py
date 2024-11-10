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
from light_gcn import LightGCNModel
# 加载 MovieLens 数据集
def load_movielens_100k(path):
    df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df['user_id'] -= 1  # 使用户ID从0开始
    df['item_id'] -= 1  # 使物品ID从0开始
    return df

def create_edge_data(df):
    user_item_interactions = df[['user_id', 'item_id']].to_numpy().T
    user_item_rating = df['rating'].astype(float).to_numpy()
    edge_index = torch.tensor(user_item_interactions, dtype=torch.long)  # 确保类型为 long
    edge_weight = torch.tensor(user_item_rating, dtype=torch.float32)
    return edge_index, edge_weight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 设置路径
movielens_path = 'ml-100k/u.data'  # 修改为实际路径
dataset = load_movielens_100k(movielens_path)

# 准备数据
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
edge_index, edge_weight = create_edge_data(train_data)
edge_index=edge_index.to(device)
edge_weight=edge_weight.to(device)
# 创建 PyG 数据对象
data = Data(edge_index=edge_index)

# 定义 LightGCN 模型


# 参数设置
num_users = dataset['user_id'].nunique()
num_items = dataset['item_id'].nunique()
embedding_dim = 64
learning_rate = 0.01
epochs = 600
x=torch.randn((num_users+num_items, embedding_dim))
# 模型、优化器和损失函数

model = LightGCNModel(num_users, num_items, embedding_dim,initial_embedding=x).to(device)
model=model.double()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练模型
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    edge_index= edge_index.to(device)
    out = model(edge_index)
    loss = criterion(out, edge_weight.double())
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# 测试模型
model.eval()
test_edge_index, test_edge_weight = create_edge_data(test_data)
with torch.no_grad():
    out = model(test_edge_index)
    test_loss = criterion(out, test_edge_weight)
    print(f'Test Loss: {test_loss.item()}')

# 保存模型（可选）
torch.save(model.state_dict(), "lightgcn_movielens.pth")