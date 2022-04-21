#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Implementation of 'A Vectorial Semantics Approach to Personality Assesment'
By Neuman et al. 2014. Avaialble at: https://www.nature.com/articles/srep04761
** Feature Extraction Step **
'''

import argparse
import sys
import pandas as pd
from csv import reader, register_dialect, writer, field_size_limit
from operator import itemgetter
from os import getcwd, listdir, makedirs
from os.path import basename, exists, isdir, isfile, join, splitext
from statistics import mean

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__author__ = "Esteban Rissola"
__credits__ = ["Esteban Rissola"]
__version__ = "1.0.1"
__maintainer__ = "Esteban Rissola"
__email__ = "esteban.andres.rissola@usi.ch"


class personality_features:
  
  def __init__(self, data):
    self.data = data


  def extract(self):

        def load_predefined_vectors(filepath):
            vectors = {}
            with open(filepath, 'rt') as fp:
                for line in fp:
                    key, payload = line.strip().split(':')
                    words = payload.split()
                    vectors[key] = words
            return vectors


        def vsm(idx2term, term_doc_mtx, pb5_vec, pd_vec, word_embb):
        # 19 Features: 10 Personality_Big5 + 5 Personality_Disorders #
            x = np.zeros((term_doc_mtx.shape[0], 19), dtype=np.float32)
            for doc_idx, doc in enumerate(term_doc_mtx):
                print('Processing doc_idx {:d}'.format(doc_idx))
                sys.stdout.flush()
                # Personality 10-Factors #
                sim_sc_pb5 = {}
                sim_idx_pb5 = {}
                for factor, components in pb5_vec.items():
                    sim_sc_pb5[factor] = np.zeros(np.count_nonzero(doc) * len(components), dtype=np.float32)
                    sim_idx_pb5[factor] = 0

                # Personality Disorders #
                sim_sc_pd = {}
                sim_idx_pd = {}
                for pd, components in pd_vec.items():
                    sim_sc_pd[pd] = np.zeros(np.count_nonzero(doc) * len(components), dtype=np.float32)
                    sim_idx_pd[pd] = 0
            
                for t_i, w in enumerate(doc):
                    if w > 0:
                        t = idx2term[t_i]
                        compute_sim(t, w, pb5_vec, sim_sc_pb5, sim_idx_pb5, word_embb)
                        compute_sim(t, w, pd_vec, sim_sc_pd, sim_idx_pd, word_embb)
            
                for ft_idx, (factor, sim) in enumerate(sim_sc_pb5.items()):
                    # Get last index (Since words may not be present in the word
                    # embeddings, consequently sim_sc_pb5 might be smaller) #
                    idx = sim_idx_pb5[factor]
                    x[doc_idx][ft_idx] = sim[:idx].mean()

                for ft_idx, (pd, sim) in enumerate(sim_sc_pd.items(), start=10):
                    # Get last index (Since all the words may not be present in the word
                    # embeddings, consequently sim_sc_pb5 might be smaller) #
                    idx = sim_idx_pd[pd]
                    x[doc_idx][ft_idx] = sim[:idx].mean()
            return x

        def compute_sim(term, weight, vec, sim_sc_vec, sim_idx_vec, word_embb):
            dim = word_embb.vector_size
            for factor, components in vec.items():
                # Get current index #
                idx = sim_idx_vec[factor]
                for word in components:
                    # Retrieve the corresponding embeddings #
                    if (word in word_embb) and (term in word_embb):
                        v_a = word_embb[word].reshape(1, dim)
                        v_b = word_embb[term].reshape(1, dim)
                        # Compute cosine similarity #
                        sim = weight * cosine_similarity(v_a, v_b)[0][0]
                        sim_sc_vec[factor][idx] = sim
                        idx += 1
                # Update index #
                sim_idx_vec[factor] = idx

        corpus = []
        users = []
        path = "./_data_/"+str(self.data)
        X = pd.read_pickle(path)
        users = X.user_name
        corpus = X.text_cleaned 
        vectorizer = CountVectorizer()
        # In case you are interested in trying other types other weighting schemes
        # vectorizer = TfidfVectorizer()
        term_doc_mtx = vectorizer.fit_transform(corpus).toarray()
        idx2term = vectorizer.get_feature_names()
        print(' -- Term-Doc Matrix Created --')

        # Load Personality (big-5) Vectors #
        filepath = "./CheckerOrSpreader/features/estepan_personality/word_vectors/pb5.vec"
        pb5_vec = load_predefined_vectors(filepath)

        # Load Personality Disorders Vectors #
        filepath = "./CheckerOrSpreader/features/estepan_personality/word_vectors/pd.vec"
        pd_vec = load_predefined_vectors(filepath)

        # Load word embeddings #
        word_embb = KeyedVectors.load_word2vec_format("./_data_/GoogleNews-vectors-negative300.bin", binary=True)
        print(' -- Word Embeddings loaded --')
        
        # For each row of the document-term matrix: #
        # * For each term t_i in document d_j #
        # * Compute sim(t_i, v_k) where v is a word in pb5_vector/pd_vector #
        # * Compute the average of the similarities obtained #

        x = vsm(idx2term, term_doc_mtx, pb5_vec, pd_vec, word_embb)
        # Replace nan with zero #
        np.nan_to_num(x, copy=False)
        filepath = "./_features_/{}/personality_scores_{}.npy".format(str(self.data),str(self.data))
        np.save(filepath, x)

        filepath = join("./_features_/", 'users_%s.txt' % str(self.data))
        with open(filepath, 'w') as fp:
           for username in users:
                fp.write('%s\n' % username)

        return 0


  #if __name__ == '__main__':
   #   extract()