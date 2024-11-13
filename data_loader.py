import pandas as pd

class Data_loader():
    def __init__(self, version):
        self.version=version
        self.sep='::'
        self.names=['userId','movieId','rating','timestemp'];            
        self.path_for_whole='./ml-1m/ratings.dat'
        self.path_for_train='./ml-1m/train_1m%s.dat'%(version)
        self.path_for_test='./ml-1m/test_1m%s.dat'%(version)
        self.num_u=6040; self.num_v=3952
        

    def data_load(self):
        
        self.whole_=pd.read_csv(self.path_for_whole, names = self.names, sep=self.sep, engine='python').drop('timestemp',axis=1).sample(frac=1,replace=False,random_state=self.version)
        self.train_set = pd.read_csv(self.path_for_train,engine='python',names=self.names).drop('timestemp',axis=1)
        self.test_set = pd.read_csv(self.path_for_test,engine='python',names=self.names).drop('timestemp',axis=1)            
                
    
        
        return self.train_set, self.test_set
    
    