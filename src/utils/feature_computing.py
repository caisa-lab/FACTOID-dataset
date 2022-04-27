import numpy as np
import pickle as pkl
import glob
import pandas as pd
import os
import torch


"""
This class bundles the different methodologies for extracting embeddings
"""
class Embedder:
    def __init__(self, embeddings_dir, embeddings_type='bert', dim=768, read_personality=True) -> None:
        self.users_embeddings = {}
        self.dim = dim
                
        if embeddings_type == 'bert':
            files = glob.glob(os.path.join(embeddings_dir[0], '*.pkl'))
            files = sorted(files, key= lambda file: int(file.split('.')[-2].split('_')[-1]))

            for index, file in enumerate(files):
                temp_embeddings = pkl.load(open(file, 'rb'))
                for user_id, embedding in temp_embeddings.items():
                    if user_id != 'desc':
                        user_embedding = self.users_embeddings.get(user_id, [])
                        
                        if len(user_embedding) < index and len(user_embedding) > 0:
                            current = user_embedding[-1]
                        elif len(user_embedding) < index and len(user_embedding) == 0:
                            current = torch.rand(dim)
                        
                        while len(user_embedding) < index:
                            user_embedding.append(current)
                        
                        user_embedding.append(torch.tensor(embedding))
                        self.users_embeddings[user_id] = user_embedding
            
        if 'usr2vec' in embeddings_type:
            files = glob.glob(os.path.join(embeddings_dir[0], '*.txt'))
            files = sorted(files, key= lambda file: int(file.split('.')[-2].split('_')[-1]))

            for index, file in enumerate(files):
                with open(file) as f:
                    next(f)
                    for line in f:
                        values = line.split(' ')
                        user_id = values[0]
                        embedding = np.array(values[-200:]).astype(np.double)
                        user_embedding = self.users_embeddings.get(user_id, [])
                       
                        if len(user_embedding) < index and len(user_embedding) > 0:
                            current = user_embedding[-1]
                        elif len(user_embedding) < index and len(user_embedding) == 0:
                            current = torch.rand(dim)
                    
                        while len(user_embedding) < index:
                            user_embedding.append(current)
                    
                        user_embedding.append(torch.tensor(embedding))
                        self.users_embeddings[user_id] = user_embedding     
            
        
        if 'usr2vec' in embeddings_type and 'rand' in embeddings_type:
            for user, embedding in self.users_embeddings.items():
                self.users_embeddings[user]  = [torch.cat((embedding[0], torch.rand(83)))]
                    
                    
        if 'usr2vec' in embeddings_type and 'liwc' in embeddings_type:
            liwc_embeddings = {}
            liwc_frame = pd.read_pickle(os.path.join(embeddings_dir[1], 'new_static_LIWC_features.pkl'))
            for index, row in liwc_frame.iterrows():
                liwc_embeddings[index] = torch.tensor(row.values)
            
            if read_personality:
                personality_frame = pd.read_pickle(os.path.join(embeddings_dir[1], 'new_static_personality_features.pkl'))
                for index, row in personality_frame.iterrows():
                    v = liwc_embeddings[index]
                    liwc_embeddings[index] = [torch.cat((v, torch.tensor(row.values)))]
            
            for user, embedding in self.users_embeddings.items():
                if user in liwc_embeddings:
                    self.users_embeddings[user]  = [torch.cat((embedding[0], liwc_embeddings[user][0]))]
                else:
                    self.users_embeddings[user]  = [torch.cat((embedding[0], torch.rand(83)))]

        if embeddings_type == 'liwc':
            liwc_frame = pd.read_pickle(os.path.join(embeddings_dir[0], 'new_static_LIWC_features.pkl'))
            for index, row in liwc_frame.iterrows():
                self.users_embeddings[index] = torch.tensor(row.values)
            
            if read_personality:
                personality_frame = pd.read_pickle(os.path.join(embeddings_dir[0], 'new_static_personality_features.pkl'))
                for user, embedding in self.users_embeddings.items():
                    if user in personality_frame.index:
                        value = personality_frame.loc[user].values
                        self.users_embeddings[user] = [torch.cat((embedding, torch.tensor(value)))]
                    else:
                        self.users_embeddings[user] = [torch.cat((embedding, torch.rand(19)))]
            
        for user, values in self.users_embeddings.items():
            while len(values) < 16:
                values.append(torch.zeros(dim))
        
        self.test_users = []
        for user, values in self.users_embeddings.items():
            if torch.equal(values[-4].double(), torch.zeros(dim).double()) and torch.equal(values[-3].double(),torch.zeros(dim).double()) \
                and torch.equal(values[-2].double(), torch.zeros(dim).double()) and torch.equal(values[-1].double(), torch.zeros(dim).double()):
                pass
            else:
                self.test_users.append(user)    
    
    def embed_user(self, idx, time_bucket=None, mode='avg'):
        if time_bucket is None:
            if idx in self.users_embeddings:
                return torch.stack(self.users_embeddings[idx]).mean(axis=0)
            else:
                return torch.rand(self.dim)
        else:
            return self.users_embeddings[idx][time_bucket]
    
    def __del__(self):
        self.users_embeddings = {}


