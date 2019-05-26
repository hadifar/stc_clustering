# -*- coding: utf-8 -*-

import os
from collections import Counter

import nltk
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


def load_stackoverflow(data_path='/Users/mac/PycharmProjects/clustering/data/Stackoverflow/rawText/'):

    # load saved features
    if os.path.isfile(data_path + 'XX.npy'):
        XX = np.load(data_path + 'XX.npy')
        y = np.load(data_path + 'y.npy')
        return XX, y

    # load SO embedding
    with open(data_path + 'vocab_withIdx.dic', 'r') as inp_indx, \
            open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
            open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        emb_index = inp_dic.readlines()
        emb_vec = inp_vec.readlines()
        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec):
            word = index_word[index.replace('\n', '')]
            word_vectors[word] = np.array(list((map(float, vec.split()))))

        del emb_index
        del emb_vec

    with open(data_path + 'title_StackOverflow.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    pca = PCA(n_components=1)
    pca.fit(all_vector_representation)
    pca = pca.components_

    XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

    XX = XX1

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    with open(data_path + 'label_StackOverflow.txt') as label_file:
        y = np.array(list((map(int, label_file.readlines()))))
        print(y.dtype)

    # save
    np.save(data_path + 'XX.npy', XX)
    np.save(data_path + 'y.npy', y)

    return XX, y


def load_search_snippet2(data_path='/Users/mac/PycharmProjects/clustering/data/SearchSnippets/new/'):
    mat = scipy.io.loadmat(data_path + 'SearchSnippets-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'SearchSnippets_vocab2idx.dic', 'r') as inp_indx:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'SearchSnippets.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        all_lines = [line for line in all_lines]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(12340, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_

    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def load_biomedical(data_path='/Users/mac/PycharmProjects/clustering/data/Biomedical/'):
    mat = scipy.io.loadmat(data_path + 'Biomedical-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'Biomedical_vocab2idx.dic', 'r') as inp_indx:
        # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
        # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'Biomedical.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        # print(sum([len(line.split()) for line in all_lines])/20000) #avg length
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, random_state=rand_seed, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_
    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def load_data(dataset_name):
    print('load data')
    if dataset_name == 'stackoverflow':
        return load_stackoverflow()
    elif dataset_name == 'biomedical':
        return load_biomedical()
    elif dataset_name == 'search_snippets':
        return load_search_snippet2()
    else:
        raise Exception('dataset not found...')
