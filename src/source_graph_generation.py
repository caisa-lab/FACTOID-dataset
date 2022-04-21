from data_collection.reddit_user_dataset import RedditUserDataset
from classification.feature_computing import Embedder
from utils.file_sort import path_sort
import os
import datetime
import time
import pickle as pkl
from os import listdir
from os.path import isfile, join
from argparse import ArgumentParser
import argparse
import gzip
import sys
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser()
parser.add_argument("--gen_source_graphs", dest="gen_source_graphs", default=False,
                    help="Whether to create noe source graphs for this setup", type=str2bool)
parser.add_argument("--source_threshold", dest="source_threshold", default=0.75,
                    help="Simmilarity threshold of the source graphs", type=float)
parser.add_argument("--embed_mode", dest="embed_mode", default='avg')
parser.add_argument("--embed_type", dest="embed_type", default='bert')
parser.add_argument("--dim", dest="dim", default=768, type=int)
parser.add_argument("--graph_type", dest="graph_type", default='linguistic')
parser.add_argument("--user_interaction_file_dir", dest='user_interaction_file_dir', default=None, required=False)
parser.add_argument("--similarity_metric", dest="similarity_metric", default="cosine_similarity")
parser.add_argument("--delta_days", dest="delta_days", default=30, type=int)
parser.add_argument("--offset_days", dest="offset_days", default=30, type=int)
parser.add_argument("--path", dest="path", required=True)
parser.add_argument("--base_dataset", dest="base_dataset_path", required=True)
parser.add_argument("--merge_liwc", dest="merge_liwc", default=False, type=str2bool)
parser.add_argument("--doc_embedding_file_path", dest="doc_embedding_file_path", required=True, default=None)
parser.add_argument("--doc_embedding_file_header", dest="doc_embedding_file_header", required=False, default='embedding_file')

args = parser.parse_args()

if args.graph_type == 'social' and args.user_interaction_file_dir is None:
    raise Exception('Need to provide a user_interaction_file_dir for the social graph')

generate_source_graphs = args.gen_source_graphs
source_threshold = args.source_threshold
embed_mode = args.embed_mode
embed_type = args.embed_type
path = args.path
source_graph_path = os.path.join(path, 'source/')
delta_days = args.delta_days
offset_days = args.offset_days
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2021, 4, 30)
merge_liwc = args.merge_liwc
embeddings_filepath = [args.doc_embedding_file_path]
dim = args.dim

if merge_liwc:
    print("Merging with LIWC")
    embeddings_filepath.append('../data/embeddings/psycho/' )
    dim = dim + 83

embedder = Embedder(embeddings_filepath, embed_type, dim=dim)
    
if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(source_graph_path):
    os.makedirs(source_graph_path)


# Generate timeframes
curr_date = start_date
timeframes = []
while curr_date + datetime.timedelta(days=delta_days) < end_date:
    print(curr_date)
    timeframes.append((curr_date, curr_date + datetime.timedelta(days=delta_days)))
    curr_date = curr_date + datetime.timedelta(days=offset_days)

print(timeframes)

base_dataset = RedditUserDataset.load_from_file(args.base_dataset_path, compression='gzip')

headers = base_dataset.data_frame.columns

print(base_dataset.data_frame)

if generate_source_graphs:
    print("Generating timeframed embeddings")
    if args.graph_type == 'social':
        base_dataset.cache_social_graph(args.user_interaction_file_dir)
    for time_index, tf in tqdm(enumerate(timeframes)):
        print("Generating source timeframe...")
        start = time.time()
        if args.graph_type == 'linguistic':
            #framed = base_dataset.timeframed_documents(tf, inplace=False)
            framed = RedditUserDataset(base_dataset.data_frame.drop(columns=['documents', 'embedding_file', 'annotation', 'bias_counter', 'factual_counter'], errors='ignore'))
            framed.generate_similarity_triplet_list(embedder, source_threshold, embed_mode,time_index,
                                                    similarity_metric=args.similarity_metric)
        elif args.graph_type == 'social':
            # Format ('2020-09-01', '2020-09-30')
            formatted_timeframe = (tf[0], tf[1])
            print(formatted_timeframe)
            framed = base_dataset.load_social_graph_from_cache(formatted_timeframe, inplace=False)
            framed = RedditUserDataset(base_dataset.data_frame.drop(columns=['documents', 'embedding_file', 'annotation', 'bias_counter', 'factual_counter'], errors='ignore'))
        else:
            raise Exception("Invalid graph type")
        framed.store_instance_to_file(source_graph_path + 'source_graph_' + str(time_index) + '.pkl')
        end = time.time()
        print("Elapsed time:" + str(end - start))

    if args.graph_type == 'linguistic':
        descriptor = {
            "graph_type": args.graph_type,
            "base_dataset": args.base_dataset_path,
            "delta_days": delta_days,
            "offset_days": offset_days,
            "embed_mode": embed_mode,
            "similarity_metric": args.similarity_metric,
            "source_threshold": source_threshold,
            "timeframes": timeframes,
            "embedding_file_path": args.doc_embedding_file_path, 
            "embedding_file_header": args.doc_embedding_file_header, 
            "embed_type": embed_type, 
            "dim": dim,
            "merge_liwc": merge_liwc}
    elif args.graph_type == 'social':
        descriptor = {
            "graph_type": args.graph_type,
            "base_dataset": args.base_dataset_path,
            "delta_days": delta_days,
            "offset_days": offset_days,
            "timeframes": timeframes,
            "embedding_file_path": args.doc_embedding_file_path,
            "embed_mode": embed_mode,
            "user_interaction_file_dir": args.user_interaction_file_dir,
            "embedding_file_header": args.doc_embedding_file_header, 
            "embed_type": embed_type, 
            "dim": dim, 
            "merge_liwc": merge_liwc}
    else:
        raise Exception("Invalid graph type")

    pkl.dump(descriptor, gzip.open(os.path.join(source_graph_path, 'source_graph_descriptor.data'), 'wb'))

timeframed_dataset = []

if args.graph_type == 'social':
    sys.exit()

for graph in path_sort(
        [join(source_graph_path, f) for f in listdir(source_graph_path) if isfile(join(source_graph_path, f))]):
    # Skip descriptor
    if "source_graph_descriptor.data" in graph:
        continue
    print("Loading source timeframes")
    print(graph)
    timeframed_dataset.append(RedditUserDataset.load_from_instance_file(graph))
