import json
import os
import time
import torch
import pandas as pd
import re



num_user=6040
num_movie=3952
model=torch.load('model.pt')
emb=model.aggregate()
emb_u, emb_v = torch.split(emb,[num_user,num_movie])


def get_rating(user_id,pos):
    _u=emb_u[user_id-1]
    _v=emb_v[pos-1]
    _u=torch.unsqueeze(_u,0)
    _v=torch.unsqueeze(_v,0)
    r=(_u*_v).sum()
    r=min(r,4)
    r=max(r,5)
    print(r)
    return r


edges=[]
with open('aug_output.jsonl') as file:
    lines=file.readlines()
    for line in lines:
        request=dict(json.loads(line))
        print(request)
        user_id=int(request['custom_id'])
        content=request['response']['body']['choices'][0]['message']['content']
        x= re.findall(r'\d+', content)
        if(len(x)!=2):
            continue
        pos,neg=x
       
        pos=int(pos)
        neg=int(neg)
        if(pos>num_movie):
            continue

        rating=get_rating(user_id,pos)
        edges.append([user_id,pos,rating])
        print(user_id,pos,rating)
        print('\n')
edges=pd.DataFrame(edges,columns=['userId','movieId','rating'])
torch.save(edges,'augmented_edges.pt')
