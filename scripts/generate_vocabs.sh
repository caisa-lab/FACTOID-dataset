#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=/home/plepi/FACTOID-dataset/logs/log.out
#SBATCH --error=/home/plepi/FACTOID-dataset/logs/log.err
#SBATCH --mail-user=joan.plepi@uni-marburg.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1GB
#SBATCH --partition=owner_fb12
#SBATCH --gres=gpu:0
#SBATCH --time=1:00:00

source /home/plepi/anaconda3/etc/profile.d/conda.sh
conda activate factoid
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/plepi/anaconda3/lib/
python ../src/create_vocabs_per_month.py --base_dataset='../data/reddit_dataset/testPipeline_factoid_20u.gzip'