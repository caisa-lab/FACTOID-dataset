#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/home/plepi/FACTOID-dataset/logs/log.out
#SBATCH --error=/home/plepi/FACTOID-dataset/logs/log.err
#SBATCH --mail-user=joan.plepi@uni-marburg.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=owner_fb12
#SBATCH --gres=gpu:0
#SBATCH --time=8:00:00

source /home/plepi/anaconda3/etc/profile.d/conda.sh
conda activate factoid
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/plepi/anaconda3/lib/

python ../src/source_graph_generation.py \
--gen_source_graphs=True \
--path='../data/reddit_dataset/linguistic/cosine/avg/bert_embeddings/' \
--base_dataset='../data/reddit_dataset/testPipeline_factoid_20u.gzip' \
--doc_embedding_file_path='../data/embeddings/bert/' \
--embed_type='bert' \
--merge_liwc='false' \
--dim=768 \
--embed_mode='avg'
