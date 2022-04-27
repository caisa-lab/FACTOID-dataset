from utils.feature_computing import *
from dataset.reddit_user_dataset import RedditUserDataset
import os
import pickle as pkl
from argparse import ArgumentParser
from tqdm import tqdm
import glob
from constants import *
from sentence_transformers import SentenceTransformer
from utils.train_utils import process_tweet


def sentence_embeddings_model(bert_model):
    model = SentenceTransformer(bert_model).to(DEVICE)
    
    return model

parser = ArgumentParser()
parser.add_argument("--vocabs_dir", required=True, type=str)
parser.add_argument("--base_dataset", dest="base_dataset_path", default='../data/reddit_dataset/reddit_corpus_balanced_filtered.gzip', type=str)
parser.add_argument("--output_dir", default="../data/reddit_dataset/bert_embeddings/", type=str)


""" 
Script to generate bert embeddings per month. 
Args:
    vocabs_dir: path to directory with users vocabularies per month. Each file, needs to contain the vocabulary for each user
                in the following format:
                    user_1 \t text_1
                    user_1 \t text_2
                The files need to named user_vocab_0.txt, user_vocab_1.txt etc.. where 0 corresponds to January 2020, 
                1 to February 2020 and so on. This is done in order to sort the files (see below)
"""
if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print("Creating directory {}".format(args.output_dir))
        os.mkdir(args.output_dir)

    base_dataset = RedditUserDataset.load_from_file(args.base_dataset_path, compression='gzip')
    model = sentence_embeddings_model('all-mpnet-base-v2')
    
    files = glob.glob(os.path.join(args.vocabs_dir, '*.txt'))
    files = sorted(files, key= lambda file: int(file.split('.')[-2].split('_')[-1]))

    for i, file in tqdm(enumerate(files), desc="File Processing Progress"):
        user_texts = {}
        embeddings = {'desc': 'Bert Embeddings per months'}

        with open(file, 'r') as f:
            for line in f:
                temp = line.split('\t')
                assert len(temp) == 2, print(line)
                user = temp[0]
                text = temp[1]
                
                texts = user_texts.get(user, [])
                texts.append(text.strip())
                user_texts[user] = texts
        
        for user_id, texts in tqdm(user_texts.items(), desc="Embedding Progress"):
            texts = [process_tweet(text) for text in texts]
            output = model.encode(texts)
            embeddings[user_id] = torch.tensor(np.mean(output, axis=0))

        pkl.dump(embeddings, open(os.path.join(args.output_dir, f'user_embeddings_{i}.pkl'), 'wb'))
