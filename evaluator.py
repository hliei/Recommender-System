
import numpy as np
import torch
from copy import deepcopy


class evaluator():
    def __init__(self,data_class,reco,N=[5,10,15, 20]):

        self.reco = reco
        self.data = data_class
        self.N = np.array(N)
        self.threshold = 3.5 # to generate ground truth set
        
        all_items = set(np.arange(1,data_class.num_v + 1))
        tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
        no_items = all_items - tot_items
        tot_items = torch.tensor(list(tot_items)) - 1
        self.no_item = (torch.tensor(list(no_items)) - 1).long().numpy()
        
        self.__gen_ground_truth_set()
        
        
    
    def __gen_ground_truth_set(self):
        self.GT = dict() # a dict storing the ground truth set for each user
        temp = deepcopy(self.data.test)
        temp = temp[temp['rating']>self.threshold].values[:,:-1]-1 # get the userId and movieId with intrest edge of the test set
        for j in range(self.data.num_u):
            if len(temp[temp[:,0]==j][:,1])>0: 
                self.GT[j] = temp[temp[:,0]==j][:,1]

        
    def precision_and_recall(self):     
        maxn = max(self.N)
        self.p = np.zeros(maxn)
        self.r = np.zeros(maxn)
        leng = 0
        maxn = max(self.N)

        for uid in self.GT:
            leng += 1
            hit_ = np.cumsum([1.0 if item in self.GT[uid] else 0.0 for idx, item in enumerate(self.reco[uid][:maxn])])
            self.p += hit_ / np.arange(1, maxn + 1)
            self.r += hit_ / len(self.GT[uid])

        self.p /= leng
        self.r /= leng