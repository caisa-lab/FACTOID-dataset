import pandas as pd
from .fake_news_detection import EvaluatedUser
from .fake_news_detection import LinkDetector
from utils.train_utils import get_embeddings_dict_from_path
import torch as torch
import networkx as nx
import pickle as pkl
import math
from datetime import datetime
from scipy.stats import pearsonr
import csv
import random
import numpy as np
import tqdm
import os
from tqdm import tqdm


class RedditUserDataset(object):

    def __init__(self, pd_frame):
        self.data_frame = pd_frame.dropna(axis=1, how='all')

    def cache_social_graph(self, social_interaction_folder):
        uids = []
        for _, row in self.data_frame.iterrows():
            uids.append(row['user_id'] )
        interaction_cache = {}
        amount = 0
        for _, row in tqdm(self.data_frame.iterrows()):
            user_file = open(os.path.join(social_interaction_folder, row['user_id'] + ".txt"), 'rb')
            user_social_interactions = []
            for interaction_str in user_file.readlines():
                interacted_user, post_timestamp, source_post_timestamp = interaction_from_str(interaction_str)
                if interacted_user in uids:
                    user_social_interactions.append((interacted_user, post_timestamp, source_post_timestamp))
            interaction_cache[row['user_id']] = user_social_interactions
            amount += len(user_social_interactions)
        print('Interaction amount:')
        print(amount)
        self.interaction_cache = interaction_cache

    def load_social_graph_from_cache(self, timeframe, inplace=False):
        if not hasattr(self, 'interaction_cache'):
            raise Exception('No interaction cache loaded!')
        uids = []
        for _, row in self.data_frame.iterrows():
            uids.append(row['user_id'])
        new_col = []
        for _, row in tqdm(self.data_frame.iterrows()):
            interactions = self.interaction_cache[row['user_id']]
            user_social_interactions = {}
            for interaction in interactions:
                if is_in_timeframe(interaction[1], timeframe) and is_in_timeframe(interaction[2], timeframe):
                    if interaction[0] in user_social_interactions.keys():
                        user_social_interactions[interaction[0]] += 1
                    else:
                        user_social_interactions[interaction[0]] = 1
            new_col.append(user_social_interactions)

        if inplace:
            self.data_frame['social_graph'] = new_col
        else:
            new_df = self.data_frame.copy(deep=True)
            new_df['social_graph'] = new_col
            return RedditUserDataset(new_df)


    @classmethod
    def load_from_file(cls, path: str, compression='infer'):
        return cls(pd.read_pickle(path, compression=compression))

    @classmethod
    def load_twitter_data(cls, label_file, crawl_folder):
        user_ids = []
        user_names = {}
        user_labels = {}
        user_docs = {}
        with open(label_file) as f:
            for user in f.readlines():
                user = user.strip()
                triple = user.split(',')
                docs = []
                try:
                    with open(crawl_folder + triple[1] + '.csv') as csvfile:
                        reader = csv.reader(csvfile, delimiter=';', quotechar="\"")
                        for row in reader:
                            if len(row) < 5:
                                continue
                            docs.append((row[1], row[2], row[3], row[4]))
                    user_docs[triple[0]] = docs
                except Exception as e:
                    print('Skipping user ' + user)
                    continue
                user_ids.append(triple[0])
                user_names[triple[0]] = triple[1]
                user_labels[triple[0]] = triple[2]
        frame = pd.DataFrame([[uid, user_labels[uid], user_docs[uid]] for uid in user_ids],
                             columns=['user_id', 'fake_news_spreader', 'documents'])
        frame.set_index('user_id', drop=False, inplace=True)
        print(frame.head())
        return cls(frame)

    def store_to_file(self, path: str):
        self.data_frame.to_pickle(path, compression='gzip')

    def store_instance_to_file(self, path: str):
        pkl.dump(self, open(path, 'wb'))

    @staticmethod
    def load_from_instance_file(path: str):
        return pd.read_pickle(open(path, 'rb'))

    def generate_balanced_training_dataset(self):
        small_df = build_balanced_fn_frame(self.data_frame)
        self.data_frame = small_df

    def filter_user_ids(self, user_ids, inplace=True):
        if inplace:
            self.data_frame = self.data_frame[self.data_frame.user_id.isin(user_ids)]
        else:
            new_df = self.data_frame.copy(deep=True)
            new_instance = RedditUserDataset(new_df[new_df.user_id.isin(user_ids)])
            if hasattr(self, "similarity_triplets"):
                new_instance.similarity_triplets = self.similarity_triplets
            return new_instance

    def build_graph_column_precomputed(self, threshold=0.75, percentage=0.05, inplace=False):
        new_df = build_linguistic_graph_precomputed(self.data_frame, self.similarity_triplets, threshold=threshold,
                                                    percentage=percentage)
        if inplace:
            self.data_frame = new_df
        else:
            new_df = new_df.copy(deep=True)
            return RedditUserDataset(new_df)

    def build_graph_column(self, graph='linguistic', mode='threshold_percentage', connect_n=0.9, percentage=0.05,
                           embed_mode='avg', inplace=False):
        raise Exception('Deprecated code')
        if graph == 'linguistic':
            df = build_linguistic_graph(self.data_frame, connect_n=connect_n, mode=mode,
                                        percentage=percentage, embed_mode=embed_mode)
        if graph == 'social':
            raise Exception("Social graph can only be build when fetching from the database")
        if inplace:
            self.data_frame = df
        else:
            return RedditUserDataset(df)

    def fit_label_amount(self, label, label_amount_map):
        collector = []
        amount_column = []
        for _, row in self.data_frame.iterrows():
            amount_column.append(len(row['documents']))

        self.data_frame['num_docs'] = amount_column

        for class_name, amount in label_amount_map.items():
            print(class_name)
            # if class_name == 0:
            #    class_filter = (self.data_frame[label] == class_name) & (self.data_frame.num_docs >= 100)
            # else:
            class_filter = (self.data_frame[label] == class_name)
            filtered_rows = self.data_frame[class_filter]
            print(len(filtered_rows))
            filtered_rows = filtered_rows.sample(n=amount)
            collector.append(filtered_rows)
        joined_frame = pd.concat(collector)
        joined_frame = joined_frame.sample(frac=1)
        self.data_frame = joined_frame

    def timeframed_documents(self, time_frame, inplace=True):
        print("Timeframe: " + str(time_frame))
        doc_col = []
        for index, row in self.data_frame.iterrows():
            user_doc_tuples = row['documents']
            docs_to_keep = []
            for doc_tuple in user_doc_tuples:
                post_date = doc_tuple[2]
                if isinstance(post_date, str):
                    post_date = datetime.strptime(doc_tuple[2], '%Y-%m-%d %H:%M:%S')
                try:
                    if time_frame[0] <= post_date.date() < time_frame[1]:
                        docs_to_keep.append(doc_tuple)
                except:
                    continue
            doc_col.append(docs_to_keep)

        if inplace:
            self.data_frame['documents'] = doc_col

            amount_column = []
            post_map = {}
            for index, row in self.data_frame.iterrows():
                amount_column.append(len(row['documents']))
                post_map[row['user_id']] = row['documents']
            tf_gt, doc_amounts = generate_ground_truth(post_map, content_index=1)
            link_amount_col = []
            for index, row in self.data_frame.iterrows():
                link_amount_col.append(doc_amounts[row['user_id']])
            self.data_frame['num_docs'] = amount_column
            self.data_frame['num_links'] = link_amount_col
        else:
            new_df = self.data_frame.copy(deep=True)
            new_df['documents'] = doc_col

            amount_column = []
            post_map = {}
            for index, row in new_df.iterrows():
                amount_column.append(len(row['documents']))
                post_map[row['user_id']] = row['documents']
            tf_gt, doc_amounts = generate_ground_truth(post_map, content_index=1)
            link_amount_col = []
            for index, row in new_df.iterrows():
                link_amount_col.append(doc_amounts[row['user_id']])
            new_df['num_docs'] = amount_column
            new_df['num_links'] = link_amount_col
            new_instance = RedditUserDataset(new_df)
            if hasattr(self, "similarity_triplets"):
                new_instance.similarity_triplets = self.similarity_triplets
            return new_instance

    def reannotate_dataset(self):
        posts_map = {}
        for index, row in self.data_frame.iterrows():
            posts_map[row['user_id']] = row['documents']
        ground_truth, amounts = generate_ground_truth(posts_map, content_index=1)
        ground_truth_column = []
        amounts_column = []
        for index, row in self.data_frame.iterrows():
            ground_truth_column.append(ground_truth[row['user_id']])
            amounts_column.append(amounts[row['user_id']])
        self.data_frame['amounts'] = amounts_column
        self.data_frame['fake_news_spreader'] = ground_truth_column

    def generate_similarity_triplet_list(self, embedder, threshold, embed_mode, time_index,
                                         similarity_metric="cosine_similarity"):
        if similarity_metric == "cosine_similarity":
            res = generate_similarity_matrix(embedder, self.data_frame, threshold, embed_mode=embed_mode, time_index=time_index,
                                             sim_func=cos_sim)
        elif similarity_metric == "pearson_similarity":
            res = generate_similarity_matrix(embedder, self.data_frame, threshold, embed_mode=embed_mode, time_index=time_index,
                                             sim_func=pearson_sim)
        else:
            raise Exception("Invalid similarity metric!")
        self.similarity_triplets = res

    def shorten_similarity_triplet_list(self, threshold):
        if hasattr(self, 'similarity_triplets'):
            new_triplets = []
            for triplet in self.similarity_triplets:
                if triplet[2] >= threshold:
                    new_triplets.append(triplet)
            self.similarity_triplets = new_triplets


    def compute_features(self, embedder, time_index=None, embed_mode='avg', num_features=768):
        feature_map = {}
        not_embedded_users = set()
       # group_by = self.data_frame.groupby(embedding_file_header)
       #for group_tuple in group_by:
           # print(group_tuple[0])
            #print(len(group_tuple[1]))
            #embedder = None
            #embedder = SavedPostBertEmbedder(os.path.join(embedding_file_path, group_tuple[0].replace('user', 'doc')))
        for index, row in self.data_frame.iterrows():
            # if len(row['documents']) == 0:
            #     feature_map[index] = torch.zeros(num_features).float()
            #     continue
            # else:
            #     try:
            #         timestamp_dict = {}
            #         for doc in row['documents']:
            #             post_date = doc[2]
            #             if isinstance(post_date, str):
            #                 post_date = datetime.strptime(doc[2], '%Y-%m-%d %H:%M:%S')
            #             timestamp_dict[doc[0]] = post_date
            #         feature_map[index] = torch.tensor(
            #             embedder.embed_user([doc[0] for doc in row['documents']], mode=embed_mode,
            #                                 timestamp_map=timestamp_dict)).float()
            user_id = row['user_id']
            try:
                feature_map[user_id]  = embedder.embed_user(user_id, time_index)
            except Exception as e:
                not_embedded_users.add(user_id)
                print("Exception while embedding user " + str(index))
                print(e)
        
        return feature_map, not_embedded_users
    
    def add_embedding_file_column(self, doc_file_path):
        """ For user2vec all the values are the same
        """
        self.data_frame['user2vec'] = [doc_file_path] * len(self.data_frame)


    def add_embedding_file_column(self, new_header, doc_file_path, overwrite=False, search_for='user_embeddings'):
        headers = self.data_frame.columns

        if new_header in headers and not overwrite:
            raise Exception('New header should not match existing column header!')

        path_dict = {}

        for emb_file_name in os.listdir(doc_file_path):
            if not search_for in emb_file_name:
                print('Skipping file ' + emb_file_name)
                continue
            print('Reading file ' + emb_file_name)
            user_emb_file = os.path.join(doc_file_path, emb_file_name)
            emb_dict = get_embeddings_dict_from_path(user_emb_file)
            for user_id in emb_dict.keys():
                path_dict[user_id] = emb_file_name

        new_col = []
        for index, row in self.data_frame.iterrows():
            new_col.append(path_dict[index])

        self.data_frame[new_header] = new_col

    def annotate_on_post_level(self):
        new_doc_col = []


        fake_detector = LinkDetector(link_file_path='../data/domain_lists/fn_domains_verified')
        real_detector = LinkDetector(link_file_path='../data/domain_lists/rn_domains_verified')

        for index, row in self.data_frame.iterrows():
            user = EvaluatedUser(index, "REDDIT")
            user.own_posts = row['documents']
            f_post_map = fake_detector.candidate(user, content_index=1)
            r_post_map = real_detector.candidate(user, content_index=1)
            doc_list = []
            for doc in row['documents']:
                post_annotations = f_post_map[doc[0]]
                post_annotations.extend(r_post_map[doc[0]])
                doc_list.append(doc + tuple([post_annotations]))

            new_doc_col.append(doc_list)

        new_df = self.data_frame.copy(deep=True)
        new_df['documents'] = new_doc_col
        return new_df


class WindowGraphData(object):
    def __init__(self, n_classes, window, n_feat, n_nodes, graph_data, features, labels, user_index):
        self.n_classes = n_classes
        self.window = window
        self.n_feat = n_feat
        self.n_nodes = n_nodes
        self.graph_data = graph_data
        self.features = features
        self.labels = labels
        self.user_index = user_index

    def print_shapes(self):
        print("n_classes=" + str(self.n_classes))
        print("window=" + str(self.window))
        print("n_feat=" + str(self.n_feat))
        print("n_nodes=" + str(self.n_nodes))
        print("graph_data: ")
        [print(elem.shape) for elem in self.graph_data]
        print("features: ")
        print(self.features.shape)
        # [print(elem.shape) for elem in self.features]
        print("labels: " + str(self.labels.shape))
    
    def build_adj_matrix(self):
        self.adj = [np.zeros((self.n_nodes, self.n_nodes)) for _ in range(len(self.graph_data))]

        for g_idx, graph_data in enumerate(self.graph_data):
            src = graph_data[0]
            trg = graph_data[1]
            for i in range(len(src)):
                self.adj[g_idx][src[i], trg[i]] = 1


def convert_timeframes_to_model_input(panda_lst, precomputed_features, ground_truth, num_features):
    user_index = {}
    counter = 0
    for _, row in panda_lst[0].iterrows():
        user_index[row['user_id']] = counter
        counter += 1

    num_users = len(user_index.values())
    graph_data = []
    feature_data = []
    labels = torch.zeros(num_users).type(torch.LongTensor)

    # Generate graph data
    print("Generating graphs...")
    for frame in panda_lst:
        edges_out = []
        edges_in = []
        for _, row in frame.iterrows():
            out_nodes_map = row['social_graph']
            for edge, weight_map in out_nodes_map.items():
                if edge in user_index.keys():
                    edges_out.append(user_index[row['user_id']])
                    edges_in.append(user_index[edge])
        edges = torch.tensor([edges_out, edges_in]).long()
        graph_data.append(edges)

    # Generate feature data
    print("Generating features...")
    for counter, frame in enumerate(panda_lst):
        timeframe_features = torch.zeros([num_users, num_features])
        for index, row in frame.iterrows():
            timeframe_features[user_index[row['user_id']]] = precomputed_features[row['user_id']][counter]
        feature_data.append(timeframe_features)
    feature_data = torch.stack(feature_data)
    # Generate ground labels
    print("Generating ground truth...")
    for _, row in panda_lst[0].iterrows():
        labels[user_index[row['user_id']]] = int(ground_truth[row['user_id']])

    return WindowGraphData(2, len(panda_lst), num_features, num_users, graph_data, feature_data, labels, user_index)

def is_in_timeframe(timestamp, timeframe):
    return timeframe[0] <= timestamp.date() < timeframe[1]

def interaction_from_str(interaction_str):
    interaction_str = str(interaction_str)
    interaction_str = interaction_str.replace('\\n', '')
    interaction_str = interaction_str.replace('\'', '')
    data = interaction_str.strip().split(';')
    interacted_user = data[3]
    post_timestamp = datetime.strptime(data[2], '%Y-%m-%d %H:%M:%S')
    source_post_timestamp = datetime.strptime(data[5], '%Y-%m-%d %H:%M:%S')
    return interacted_user, post_timestamp, source_post_timestamp


def generate_ground_truth(post_map, content_index=2):
    print("Automatically annotating ground truth...")
    try:
        fake_detector = LinkDetector(link_file_path='../data/domain_lists/fn_domains_verified')
        real_detector = LinkDetector(link_file_path='../data/domain_lists/rn_domains_verified')
    except Exception as e:
        fake_detector = LinkDetector(link_file_path='data/domain_lists/fn_domains_verified')
        real_detector = LinkDetector(link_file_path='data/domain_lists/rn_domains_verified')
    res = {}
    amount_col = {}
    spreader = 0
    real_spreader = 0
    counter = 0
    for entry, posts in post_map.items():
        counter += 1
        user = EvaluatedUser(entry, "REDDIT")
        user.own_posts = posts
        fake, f_ids = fake_detector.candidate(user, content_index=content_index, min_links=1)
        real, r_ids = real_detector.candidate(user, content_index=content_index, min_links=1)
        amount_col[entry] = (len(r_ids), len(f_ids))

        if fake:
            res[entry] = 1
            spreader += 1
        elif real:
            res[entry] = 0
            real_spreader += 1
        else:
            res[entry] = -1

    print("Found " + str(spreader) + " fake news spreaders")
    print("Found " + str(real_spreader) + " real news spreaders")
    print(fake_detector.domain_counter)
    print(real_detector.domain_counter)
    return res, amount_col


def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def pearson_sim(vec1, vec2):
    val, p = pearsonr(vec1, vec2)
    return val


def generate_similarity_matrix(embedder, pd_frame, threshold, embed_mode='avg', time_index=None, sim_func=cos_sim):
    embedding_dict = {}

    for _, row in pd_frame.iterrows():
        user_id = row['user_id']

        try:
            embedding_dict[user_id] = embedder.embed_user(user_id, time_index).numpy()

        except Exception as e:
            print("Exception while embedding user " + str(row['user_id']))
            print(e)

    visited = []
    triplets = []
    for uid, embedding in embedding_dict.items():
        for comp_uid, comp_embedding in embedding_dict.items():
            if comp_uid in visited:
                continue
            if not comp_uid == uid:
                if embedding is None or comp_embedding is None:
                    continue
                sim = sim_func(embedding, comp_embedding)
                if sim > threshold:
                    triplets.append((uid, comp_uid, sim))
        visited.append(uid)
    return triplets


def build_linguistic_graph_precomputed(pd_frame, triplet_list, threshold=0.75, percentage=0.05):
    dists = {}
    ids = set(pd_frame['user_id'].values)
    for triplet in triplet_list:
        if triplet[0] not in ids or triplet[1] not in ids:
            continue
        if triplet[2] >= threshold:
            if triplet[0] in dists.keys():
                dists[triplet[0]][triplet[1]] = triplet[2]
            else:
                dists[triplet[0]] = {triplet[1]: triplet[2]}

            if triplet[1] in dists.keys():
                dists[triplet[1]][triplet[0]] = triplet[2]
            else:
                dists[triplet[1]] = {triplet[0]: triplet[2]}
    res = {}
    for user, edge_dict in dists.items():
        filtered_dists = {k: v for k, v in sorted(edge_dict.items(), key=lambda item: item[1])}
        amount = math.ceil(percentage * len(filtered_dists))
        res[user] = dict(list(filtered_dists.items())[-amount:])

    res_col = []
    for _, row in pd_frame.iterrows():
        if row['user_id'] in dists.keys():
            res_col.append(res[row['user_id']])
        else:
            res_col.append({})

    pd_frame['social_graph'] = res_col
    return pd_frame


def balance_train_mask(x, y, label_filter=None):
    label_map = {}
    for i in range(len(x)):
        if label_filter is not None and y[i] not in label_filter:
            continue
        if y[i].item() in label_map.keys():
            label_map[y[i].item()].append(i)
        else:
            label_map[y[i].item()] = [i]

    min_len = min([len(ids) for ids in label_map.values()])
    print(min_len)
    balanced_mask = []

    for ids in label_map.values():
        random.shuffle(ids)
        balanced_mask.append(ids[:min_len])

    # List of lists
    return balanced_mask


def build_balanced_fn_frame(source_frame):
    spreader_filter = source_frame['fake_news_spreader'] == 1
    non_spreader_filter = source_frame['fake_news_spreader'] == 0
    spreader = source_frame[spreader_filter]
    non_spreader = source_frame[non_spreader_filter]
    non_spreader = non_spreader.sample(n=len(spreader))
    joined_frame = pd.concat([spreader, non_spreader])
    joined_frame = joined_frame.sample(frac=1)
    return joined_frame
