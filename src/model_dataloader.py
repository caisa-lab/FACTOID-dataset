from dataset.reddit_user_dataset import RedditUserDataset
from dataset.reddit_user_dataset import convert_timeframes_to_model_input
from utils.feature_computing import Embedder
import pickle as pkl
import random
import gzip
import time
import os
from os import listdir
from utils.file_sort import path_sort
from argparse import ArgumentParser
import numpy as np
from os.path import isfile, join
import json
from utils.utils import *
import torch


parser = ArgumentParser()
parser.add_argument("--base_dataset", dest="base_dataset_path", required=True, type=str)
parser.add_argument("--source_frames", dest="source_frames_path", required=True, type=str)
parser.add_argument("--sample_dir", dest="sample_dir", required=True, type=str)
parser.add_argument("--n_users", dest="n_users", default=300, type=int)
parser.add_argument("--n_train_samples", dest="n_train_samples", default=800, type=int)
parser.add_argument("--n_val_samples", dest="n_val_samples", default=200, type=int)
parser.add_argument("--n_test_samples", dest="n_test_samples", default=1, type=int)
parser.add_argument("--threshold", dest="threshold", default=0.8, type=float)
parser.add_argument("--percentage", dest="percentage", default=1, type=float)
parser.add_argument("--root_seed", dest="root_seed", default=1337, type=int)

parser.add_argument("--train_min_index", dest="train_min_index", default=0, type=int)
parser.add_argument("--train_max_index", dest="train_max_index", default=1, type=int)
parser.add_argument("--val_min_index", dest="val_min_index", default=0, type=int)
parser.add_argument("--val_max_index", dest="val_max_index", default=1, type=int)
parser.add_argument("--test_min_index", dest="test_min_index", default=0, type=int)
parser.add_argument("--test_max_index", dest="test_max_index", default=1, type=int)

parser.add_argument("--user_ids", dest="user_ids", required=False, type=str, default="../data/reddit_dataset/user_splits/")
parser.add_argument("--random_features", dest="random_features", required=False, default=False, type=str2bool)


if __name__ == '__main__':
    args = parser.parse_args()
    base_dataset = RedditUserDataset.load_from_file(args.base_dataset_path, compression='gzip')


    # Build ground truth
    ground_truth = {}
    for _, row in base_dataset.data_frame.iterrows():
        ground_truth[row['user_id']] = row['fake_news_spreader']

    source_frame_dir = args.source_frames_path
    target_dir = args.sample_dir

    print("Running model dataloader with target dir {}".format(target_dir))

    if not os.path.exists(os.path.join(target_dir, 'train_samples/')):
        os.makedirs(os.path.join(target_dir, 'train_samples/'))
    if not os.path.exists(os.path.join(target_dir, 'val_samples/')):
        os.makedirs(os.path.join(target_dir, 'val_samples/'))
    if not os.path.exists(os.path.join(target_dir, 'test_samples/')):
        os.makedirs(os.path.join(target_dir, 'test_samples/'))

    train_min_index = args.train_min_index
    train_max_index = args.train_max_index
    val_min_index = args.val_min_index
    val_max_index = args.val_max_index
    test_min_index = args.test_min_index
    test_max_index = args.test_max_index
    n_users = args.n_users
    n_train_samples = args.n_train_samples
    n_val_samples = args.n_val_samples
    n_test_samples = args.n_test_samples
    threshold = args.threshold
    percentage = args.percentage
    ROOT_SEED = args.root_seed


    source_graph_descriptor = pkl.load(
        gzip.open(os.path.join(args.source_frames_path, 'source_graph_descriptor.data'), 'rb'))

    doc_embedding_file_path = [source_graph_descriptor['embedding_file_path']]
    embed_type = source_graph_descriptor['embed_type']
    doc_embedding_file_header = 'embedding_file'
    embed_mode = source_graph_descriptor['embed_mode']


    if 'dim' in source_graph_descriptor:
        dim = source_graph_descriptor['dim']
    else:
        dim = 768

    if 'merge_liwc' in source_graph_descriptor:
        merge_liwc = source_graph_descriptor['merge_liwc']
        if merge_liwc:
            print("Merging with LIWC")
            doc_embedding_file_path.append('../data/embeddings/psycho/' )

    dataset_descriptor = {}
    dataset_descriptor['n_users'] = n_users
    dataset_descriptor['n_train_samples'] = n_train_samples
    dataset_descriptor['n_val_samples'] = n_val_samples
    dataset_descriptor['n_test_samples'] = n_test_samples
    dataset_descriptor['threshold'] = threshold
    dataset_descriptor['percentage'] = percentage
    dataset_descriptor['root_seed'] = ROOT_SEED
    dataset_descriptor['base_dataset'] = args.base_dataset_path
    dataset_descriptor['source_frames'] = args.source_frames_path
    dataset_descriptor['embed_mode'] = embed_mode
    dataset_descriptor['time_splits'] = [[train_min_index, train_max_index],
                                        [val_min_index, val_max_index],
                                        [test_min_index, test_max_index]]
    dataset_descriptor['doc_embedding_file_path'] = doc_embedding_file_path
    dataset_descriptor['doc_embedding_file_header'] = doc_embedding_file_header
    dataset_descriptor['user_ids'] = args.user_ids

    json.dump(dataset_descriptor, open(os.path.join(target_dir, 'dataset_descriptor.json'), 'w'))

    random.seed(ROOT_SEED)
    np.random.seed(ROOT_SEED)
    sampling_seeds = [int(random.uniform(0, 1000000)) for i in range(n_train_samples + n_test_samples + n_val_samples)]

    embedder = Embedder(doc_embedding_file_path, embed_type, dim)

    timeframed_dataset = []

    for graph in path_sort(
            [join(source_frame_dir, f) for f in listdir(source_frame_dir) if isfile(join(source_frame_dir, f))]):
        if "source_graph_descriptor.data" in graph:
            continue
        print(graph)
        timeframe_ds = RedditUserDataset.load_from_instance_file(graph)
        timeframe_ds.shorten_similarity_triplet_list(threshold)
        timeframed_dataset.append(timeframe_ds)
        doc_sum = 0
        users = 0
        for index, row in RedditUserDataset.load_from_instance_file(graph).data_frame.iterrows():
            users += 1
        
        
    # Split user
    train_ids = read_userids('train_ids.txt', args.user_ids)
    val_ids = read_userids('val_ids.txt', args.user_ids)
    test_ids = read_userids('test_ids.txt', args.user_ids)
    print(len(train_ids))
    print(len(val_ids))
    print(len(test_ids))

    # Validating the splits
    for uid in train_ids:
        if uid in val_ids:
            raise Exception("Invalid split!")
        if uid in test_ids:
            raise Exception("Invalid split")

    for uid in val_ids:
        if uid in test_ids:
            raise Exception("Invalid split!")

    train_sample_frame = base_dataset.filter_user_ids(train_ids, inplace=False).data_frame
    print(len(train_sample_frame))
    val_sample_frame = base_dataset.filter_user_ids(val_ids, inplace=False).data_frame
    print(len(val_sample_frame))
    test_sample_frame = base_dataset.filter_user_ids(test_ids, inplace=False).data_frame
    print(len(test_sample_frame))


    precomputed_features = {}
    not_embedded_users = set()
    for _, row in base_dataset.data_frame.iterrows():
        precomputed_features[row['user_id']] = []

    if not args.random_features:
        for time_index, frame in enumerate(timeframed_dataset):
            print('Precomputing features for timeframe...')
            feature_map, current_not_emb_users = frame.compute_features(embedder, time_index=time_index, embed_mode=embed_mode)
            not_embedded_users.update(current_not_emb_users)

            for index, feature in feature_map.items():
                if index in precomputed_features.keys():
                    precomputed_features[index].append(feature_map[index])
    else:
        for frame in timeframed_dataset:
            print('Precomputing random features...')
            for _, row in frame.data_frame.iterrows():
                precomputed_features[row['user_id']].append(torch.tensor(np.random.uniform(low=-1.5, high=1.5, size=dim)))

    print("Could not embed {} users".format(len(not_embedded_users)))
    seed_counter = -1

    train_sample_frame = train_sample_frame[~train_sample_frame['user_id'].isin(not_embedded_users)] 
    val_sample_frame = val_sample_frame[~val_sample_frame['user_id'].isin(not_embedded_users)] 
    test_sample_frame = test_sample_frame[~test_sample_frame['user_id'].isin(not_embedded_users)] 

    start = time.time()
    for n in range(n_train_samples):
        seed_counter += 1
        sample_ids = train_sample_frame.sample(n=n_users, random_state=sampling_seeds[seed_counter])['user_id']
        sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in timeframed_dataset]

        start_frame = train_min_index
        train_window = train_max_index - train_min_index
        print(start_frame)
        print(start_frame + train_window)
        sample_frames = sample_frames[start_frame:start_frame + train_window]
        
        sample_frames = [
            tf.build_graph_column_precomputed(threshold=threshold, percentage=percentage, inplace=False).data_frame for
            _, tf in
            enumerate(sample_frames)]

        
        sample = convert_timeframes_to_model_input(sample_frames, {k: v[start_frame:start_frame + train_window] for k, v in
                                                                precomputed_features.items()}, ground_truth, dim)
        sample.print_shapes()
        pkl.dump(sample, gzip.open(os.path.join(target_dir, 'train_samples/') + 'sample_' + str(n) + '.data', 'wb'))

    end = time.time()
    print("Elapsed time:" + str(end - start))

    start = time.time()
    for n in range(n_val_samples):
        seed_counter += 1
        sample_ids = val_sample_frame.sample(n=n_users, random_state=sampling_seeds[seed_counter])['user_id']
        sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in timeframed_dataset]
        
        start_frame = val_min_index
        val_window = val_max_index - val_min_index

        print(start_frame)
        print(start_frame + val_window)

        sample_frames = sample_frames[start_frame:start_frame + val_window]

        sample_frames = [
            tf.build_graph_column_precomputed(threshold=threshold, percentage=percentage, inplace=False).data_frame for
            index, tf in
            enumerate(sample_frames)]

        sample = convert_timeframes_to_model_input(sample_frames, {k: v[start_frame:start_frame + val_window] for k, v in
                                                                precomputed_features.items()}, ground_truth, dim)
        sample.print_shapes()
        pkl.dump(sample, gzip.open(os.path.join(target_dir, 'val_samples/') + 'sample_' + str(n) + '.data', 'wb'))
    end = time.time()
    print("Elapsed time:" + str(end - start))

    start = time.time()
    sample_ids = list(set(test_sample_frame['user_id'].values).intersection(set(embedder.test_users)))

    print(f'Amount of test users {len(sample_ids)}')
    for n in range(n_test_samples):
        seed_counter += 1
        sample_frames = [frame.filter_user_ids(sample_ids, inplace=False) for frame in timeframed_dataset]

        start_frame = test_min_index
        test_window = test_max_index - test_min_index

        print(start_frame)
        print(start_frame + test_window)
        sample_frames = sample_frames[start_frame:]
        sample_frames = [
            tf.build_graph_column_precomputed(threshold=threshold, percentage=percentage, inplace=False).data_frame for
            index, tf in
            enumerate(sample_frames)]
    
            
        sample = convert_timeframes_to_model_input(sample_frames, {k: v[start_frame:] for k, v in
                                                                precomputed_features.items()}, ground_truth, dim)
        sample.print_shapes()
        pkl.dump(sample, gzip.open(os.path.join(target_dir, 'test_samples/') + 'sample_' + str(n) + '.data', 'wb'))

    end = time.time()
    print("Elapsed time:" + str(end - start))


    dataset_descriptor['source_frame_descriptor'] = source_graph_descriptor

    for key, val in dataset_descriptor.items():
        if isinstance(val, dict):
            dataset_descriptor[key] = {k: convert_value(v) for k,v in val.items()}
        else:
            dataset_descriptor[key] = val


    json.dump(dataset_descriptor, open(os.path.join(target_dir, 'dataset_descriptor.json'), 'w'))

