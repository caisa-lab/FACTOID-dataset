# FACTOID: A New Dataset for Identifying Misinformation Spreaders and Political Bias

## 1 Overview

FACTOID: a user-level **FAC**tuality and p**O**litical b**I**as **D**ataset, that contains a set of 4,150 news-spreading users with 3.3M Reddit posts in discussions on contemporary political topics, covering the time period from January 2020 to April 2021 on individual user level. You can find the dataset [here](https://drive.google.com/drive/folders/1MB6zsrhNerZQlLFBdjJ8sDbvXa2NcELZ).

## 2 Setup

### 2.1 Environment Setup

* With conda
  
    ```conda env export > environment.yml```
* With pip

    ```pip install -r requirements.txt```

## 3 Usage

### 3.1 Reddit Posts Crawling

   Crawl reddit posts using the ids provided in the dataset and fill the empty strings inside the dataframe.  

### 3.2 User Embeddings

  First extract user vocabularies 

  ```
  python create_vocabs_per_month.py --base_dataset=../data/reddit_dataset/factoid_dataset.gzip
  ```

  Then run the codes to generate
          
   * UBERT embeddings

  ```
   python user_embeddings_per_month.py --vocabs_dir='../data/user_vocabs_per_month' --base_dataset='../data/reddit_dataset/factoid_dataset.gzip'
  ```

  * [User2Vec](https://github.com/samiroid/usr2vec)
  * [Psycho Linguistic Features](https://github.com/caisa-lab/FACTOID-dataset/tree/main/src/psycho_ling_embeddings)

### 3.3 Generate Graphs and Samples

  To generate graph samples, example script. Change the parameters based on the embeddings you want to use. The argument `embed_type`  takes the following values `['bert', 'usr2vec', 'usr2vec_rand', 'usr2vec_liwc', 'liwc']`

  ```
  python source_graph_generation.py \
  --gen_source_graphs=True \
  --path='../data/reddit_dataset/linguistic/cosine/avg/bert_embeddings/' \ 
  --base_dataset='../data/reddit_dataset/factoid_dataset.gzip' \
  --doc_embedding_file_path='../data/embeddings/bert/' \
  --embed_type='bert' \
  --merge_liwc='false' \
  --dim=768 \
  --embed_mode='avg' |& tee ../logs/graph_generation.txt
  ```

  Then after creating the graph samples, run the following to make the split

  ```
  python model_dataloader.py \
  --n_users=200 \
  --n_train_samples=1000 \
  --n_val_samples=200 \
  --base_dataset='../data/reddit_dataset/factoid_dataset.gzip' \
  --source_frames='../data/reddit_dataset/linguistic/cosine/avg/bert_embeddings/source' \
  --sample_dir='../data/reddit_dataset/model_samples_avg/bert_embeddings/' \
  --user_ids='../data/reddit_dataset/user_splits/' \
  --threshold=0.8 |& tee ../logs/model_dataloader.txt
  ```

### 3.4 Training Model

  After training, validation, test samples are created, run the model using the following

  ```
  python training_graph.py --patience=40 \
  --run_id='bert_embeddings' \
  --sample_dir='../data/reddit_dataset/model_samples_avg/bert_embeddings/'  \
  --result_dir='../results/' \
  --checkpoint_dir='../results/checkpoints/' \
  --max_epochs=50 \
  --learning_rate=5e-5 \
  --nheads=4 \
  --dropout=0.2 \
  --nhid_graph=256 \
  --nhid=128 \
  --users_dim=768 \
  --gnn='gat' |& tee ../logs/graph_model_main.txt
  ```
