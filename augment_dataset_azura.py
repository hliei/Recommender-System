import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time
from promt_con import construct_prompting
from openai import AzureOpenAI
from load_dataset import load_movielens1m_data
import json
from copy import deepcopy

movielens_path = 'ml-1m_ori'  
user,movies,rating = load_movielens1m_data(base_path='ml-1m_ori')
dataset = pd.merge(rating, movies, left_on='MovieID', right_on='MovieID')
dataset = pd.merge(dataset, user, left_on='UserID', right_on='UserID')
dataset=dataset[dataset['Rating']>=4]
num_user=6040
num_movie=3952



all_items=dataset['MovieID'].unique()
for i in range(len(all_items)):
    if(all_items[i]==1736):
        sadadsa=2
    if(movies[movies['MovieID']==all_items[i]]['Title'].empty==True):
        all_items.remove(all_items[i])
def recommend(emb_u,emb_v,user_id,num_sample,user_data):

    known_items=user_data["MovieID"].unique()
    unknown_items=set(all_items)-set(known_items)
    unknown_items=list(unknown_items)
    unknown_items = [item - 1 for item in unknown_items]
    rating=torch.mm(torch.unsqueeze(emb_u[user_id-1],dim=0), emb_v.T)
    _,rec=torch.topk(rating, k=num_sample)
    rating=rating.tolist()[0]
    indexed_elements = [(index, value) for index, value in enumerate(rating)]
    
    filtered_elements = [item for item in indexed_elements if item[0]  in unknown_items]
    
    filtered_elements.sort(key=lambda x: x[1], reverse=True)
    top_k_elements = filtered_elements[:num_sample]
    
    top_k_values = [item[0]+1 for item in top_k_elements]
    
    return top_k_values

def contain_empty(num_sample,candidate):
    for j in candidate:
        #j=1618
        if movies[movies['MovieID']==j]['Title'].empty==True or movies[movies['MovieID']==j]['Genres'].empty==True:
            return True
    return False

def build_request(promt, user_id):
    request = {
        "method": "POST",
        "url": "/chat/completions",
        "custom_id": str(user_id),
        "body":{"model": "gpt-35-turbo-global","messages":[]}
    }
    request["body"]["messages"].append({"role": "system", "content": "You are an AI assistant that helps people find information."})
    request["body"]["messages"].append({"role": "user", "content": promt})
    #request=str(request)
    return request

def get_item_list(num_sample,user_id,user_data,emb_u,emb_v):
    emb_user=emb_u[user_id-1]
    
    
    _,rec=torch.topk(rating, k=num_sample)
    rec=rec.tolist()[0]
    return rec

augmented_samples = []
cnt=0


grouped_users = dataset.groupby('UserID')
attri_type=['title','unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
                                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
augmented_edge_list=[]
cnt=0
model=torch.load("model.pt")
emb = model.aggregate();
emb_u, emb_v = torch.split(emb,[num_user,num_movie])


for user_id, user_data in grouped_users:
    num_sample=5
    item_attribute = movies
    candidate_list=recommend(emb_u,emb_v,user_id,num_sample,user_data)
    sorted_user_data=deepcopy(user_data)
    sorted_user_data=sorted_user_data.sort_values(by='Rating',ascending=False)
    #item_list = user_data['MovieID'].sample(5).tolist()
    item_list=sorted_user_data["MovieID"].tolist()
    item_list=item_list[:num_sample]
    if(1736 in candidate_list):
        coassdmna=2
    #while contain_empty(num_sample,item_list):
        #item_list=movies['MovieID'].sample(5).tolist()
    #candidate_list = dataset['MovieID'].sample(5).tolist()
    promt=construct_prompting(item_attribute, item_list, candidate_list)
    request=build_request(promt, user_id)
    file_name = 'input.jsonl'

    with open(file_name, 'a',encoding='utf-8') as json_file:
        json.dump(request, json_file, ensure_ascii=False)
        json_file.write('\n')
    #print(user_id)

    print(cnt)
    cnt+=1
    
torch.save(augmented_edge_list, 'augmented_edge_list.pt')
