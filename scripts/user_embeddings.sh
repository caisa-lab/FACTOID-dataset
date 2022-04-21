#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/home/plepi/FACTOID-dataset/logs/log.out
#SBATCH --error=/home/plepi/FACTOID-dataset/logs/log.err
#SBATCH --mail-user=joan.plepi@uni-marburg.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=owner_fb12
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source /home/plepi/anaconda3/etc/profile.d/conda.sh
conda activate factoid
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/plepi/anaconda3/lib/
python ../src/user_embeddings_per_month.py --vocabs_dir='../data/user_vocabs_per_month' \
--base_dataset='../data/reddit_dataset/testPipeline_factoid_20u.gzip' \
