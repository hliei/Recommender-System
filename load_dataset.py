import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
class MovieLensDataset(Dataset):
    def __init__(self, ratings_file, users_file, movies_file):
        self.ratings = pd.read_csv(ratings_file, sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding='ISO-8859-1')
        self.users = pd.read_csv(users_file, sep='::', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding='ISO-8859-1')
        self.movies = pd.read_csv(movies_file, sep='::', names=['MovieID', 'Title', 'Genres'], engine='python', encoding='ISO-8859-1')

        
    def __len__(self):
        return len(self.ratings)
    def get_data(self):
        return self.users,self.movies,self.ratings
    def __getitem__(self, idx):
        user = self.ratings.iloc[idx, 0]
        movie = self.ratings.iloc[idx, 1]
        rating = self.ratings.iloc[idx, 2]
        return torch.tensor(user, dtype=torch.long), torch.tensor(movie, dtype=torch.long), torch.tensor(rating, dtype=torch.float)
    
def load_movielens1m_data(base_path,batch_size=64):
    ratings_file = os.path.join(base_path,'ratings.dat')
    users_file = os.path.join(base_path,'users.dat')
    movies_file = os.path.join(base_path,'movies.dat')
    
    dataset = MovieLensDataset(ratings_file, users_file, movies_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataset.get_data()

'''
user,movies,rating = load_movielens1m_data(base_path='ml-1m_ori')
user_data=rating.groupby('UserID')
print(len(rating))'''