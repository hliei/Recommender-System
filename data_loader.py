import pandas as pd

class Data_loader():
    def __init__(self):
        self.sep='::'
        self.names=['userId','movieId','rating','timestemp'];            
        self.path_for_whole='./ml-1m_ori/ratings.dat'
        self.path_for_train='./ml-1m_split/train.dat'
        self.path_for_test='./ml-1m_split/test.dat'
        self.num_u=6040; self.num_v=3952
        

    def data_load(self):
        
        self.whole_=pd.read_csv(self.path_for_whole, names = self.names, sep=self.sep, engine='python').drop('timestemp',axis=1).sample(frac=1,replace=False)
        self.train_set = pd.read_csv(self.path_for_train,engine='python',names=self.names).drop('timestemp',axis=1)
        self.test_set = pd.read_csv(self.path_for_test,engine='python',names=self.names).drop('timestemp',axis=1)            
                
    
        
        return self.train_set, self.test_set
    
    