import torch
from torch.utils.data import DataLoader
from torch import optim

from tqdm import tqdm
from evaluator import evaluator as ev
from helper_func import  bipartite_dataset, deg_dist,gen_top_k
from data_loader import Data_loader
from pngnn import PNGNN

def main(cfg):
    dataset=Data_loader(cfg['version'])
    train,test = dataset.data_load()
    train = train.astype({'userId':'int64', 'movieId':'int64'})
    dataset.train = train; dataset.test = test
    
    # Generate the degree distribution of items in train set
    neg_dist = deg_dist(train,dataset.num_v)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model= PNGNN(train, dataset.num_u,dataset.num_v,threshold=cfg['threshold'],num_layers = cfg['num_layers'],MLP_layers=cfg['MLP_layers'],dim=cfg['dim'],device=device,reg=cfg['reg'])
    model.data_p.to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = cfg['lr'])
    
    
    print("Training starts...")
    model.train()
    training_dataset=bipartite_dataset(train,neg_dist,cfg['threshold'],dataset.num_u,dataset.num_v,cfg['num_neg_samples'])
    
    for EPOCH in range(1,cfg['epoch']+1):
        LOSS=0
        total_counts=0
        #training_dataset.edge_4 = training_dataset.edge_4_tot[:,:,EPOCH%20-1]
        training_dataset.negs_gen_()
        
        ds = DataLoader(training_dataset,batch_size=cfg['batch_size'],shuffle=True)
        pbar = tqdm(desc = 'Version : {} Epoch {}/{}'.format(cfg['version'],EPOCH,cfg['epoch']),total=len(ds),position=0)
        
        for u,v,w,negs in ds:  # index of batch users, items, ratings, negative samples
            total_counts+=len(u)
            optimizer.zero_grad()
            loss = model(u,v,w,negs,device)
            loss.backward()                
            optimizer.step()
            LOSS+=loss.item() * len(ds)
            
            pbar.update(1)
            pbar.set_postfix({'loss':loss.item()})

        pbar.close()
        print('Epoch : %s, Loss : %s'%(EPOCH,LOSS/total_counts))

        if EPOCH%20 ==1:

            model.eval()
            emb = model.get_embedding()
            emb_u, emb_v = torch.split(emb,[dataset.num_u,dataset.num_v])
            emb_u = emb_u.cpu().detach(); emb_v = emb_v.cpu().detach()
            r_hat = emb_u.mm(emb_v.t())
            reco = gen_top_k(dataset,r_hat)
            eval_ = ev(dataset,reco)
            eval_.precision_and_recall()
            print(" /* Recommendation Accuracy */")
            print('K :: %s'%(eval_.N))
            print('Precision at K = [5, 10, 15, 20]:: ',eval_.p[eval_.N-1])
            print('Recall at K = [5, 10, 15, 20] :: ',eval_.r[eval_.N-1])
            model.train()

cfg = { 'version':1,
        'batch_size':1024,
        'dim':64,
        'lr':5e-3,
        'threshold':3.5,
        'num_neg_samples':40,
        'num_layers':4,
        'MLP_layers':2,
        'epoch':100,
        'reg':0.05
}

if __name__ == '__main__':
    main(cfg)
