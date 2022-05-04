import os
import random
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from tqdm import tqdm

import sys
sys.path.append('../')
from utils.utils import loadWord2Vec, clean_str, clean_str_indic



if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'hin', 'tel', 'bn', 'gu', 'kn', 'ml', 'ta','mar']
indic = ['hin', 'tel', 'bn', 'gu', 'kn', 'ml', 'mar', 'ta']

# build corpus
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
#_, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

print("Reading the metadata of ", dataset, " language")
with open('../data/' + dataset + '.txt', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
# print(doc_train_list)
# print(doc_test_list)

print("Reading the cleaned data of ", dataset, " language")
doc_content_list = []
with open('../data/corpus/' + dataset + '.clean.txt', 'r') as f:
    lines = f.readlines();
    for line in tqdm(lines):
        doc_content_list.append(line.strip())
# print(doc_content_list)

print("Generating train ids")
train_ids = []
for train_name in tqdm(doc_train_list):
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print(train_ids)
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

print("Writing ids to ", dataset, ".train.index")
train_ids_str = '\n'.join(str(index) for index in train_ids)
with open('../data/' + dataset + '.train.index', 'w') as f:
    f.write(train_ids_str)


print("Generating test ids")
test_ids = []
for test_name in tqdm(doc_test_list):
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

print("Writing ids to ", dataset, ".test.index")
test_ids_str = '\n'.join(str(index) for index in test_ids)
with open('../data/' + dataset + '.test.index', 'w') as f:
    f.write(test_ids_str)


ids = train_ids + test_ids
print(ids)
print(len(ids))

print("Generating shuffle of train and test ids")
shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

print("Writing shuffled id")
with open('../data/' + dataset + '_shuffle.txt', 'w') as f:
    f.write(shuffle_doc_name_str)

print("Writing shuffled data")
with open('../data/corpus/' + dataset + '_shuffle.txt', 'w') as f:
    f.write(shuffle_doc_words_str)


print("Building vocab")
# build vocab
word_freq = {}
word_set = set()
for doc_words in tqdm(shuffle_doc_words_list):
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)
print("Vocab size: ", vocab_size)

print("Generating word_doc list")
word_doc_list = {}

for i in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

print("Generating word-doc frequency dictionary")        
word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

print("Generating word2idx")
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

print("Writing vocab to the file")
with open('../data/corpus/' + dataset + '_vocab.txt', 'w') as f:
    f.write(vocab_str)


'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
'''

'''
Word definitions end
'''

print("Extracting labels list from metadata")
# label list
label_set = set()
for doc_meta in tqdm(shuffle_doc_name_list):
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

print("Writing labels to the file")
label_list_str = '\n'.join(label_list)
with open('../data/corpus/' + dataset + '_labels.txt', 'w') as f:
    f.write(label_list_str)


# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
val_size = int(0.05 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

with open('../data/' + dataset + '.real_train.name', 'w') as f:
    f.write(real_train_doc_names_str)

print("Generating the co-occurance matrix for train set")
row_x = []
col_x = []
data_x = []
for i in tqdm(range(real_train_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

print("Generating csr_matrix")        
# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

print("Training labels from metadata")
y = []
for i in tqdm(range(real_train_size)):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
print(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

print("Generating co-occurance matrix for test set")
row_tx = []
col_tx = []
data_tx = []
for i in tqdm(range(test_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

print("Generating csr_matrix")
# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

print("Testing labels from metadata")
ty = []
for i in tqdm(range(test_size)):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

print("Generating word-vectors")
word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in tqdm(range(len(vocab))):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []
print("All co-occurance matrix")
for i in tqdm(range(train_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

print("All labels from train")
ally = []
for i in tqdm(range(train_size)):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

print("Onehot representation for labels")
for i in tqdm(range(vocab_size)):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

print("Doc word hetrogeneous graph")
# word co-occurence with context windows
window_size = 10
windows = []

for doc_words in tqdm(shuffle_doc_words_list):
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


print("Word window frequency")
word_window_freq = {}
for window in tqdm(windows):
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

print("Word-pair count")
word_pair_count = {}
for window in tqdm(windows):
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights
print("PMI as weight")
num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# doc word frequency
doc_word_freq = {}

print("Doc-word Frequency")
for doc_id in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

print("Generating the weights")
for i in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

print("Dumping weights")
# dump objects
with open("../data/ind.{}.x".format(dataset), 'wb') as f:
    pkl.dump(x, f)

with open("../data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("../data/ind.{}.tx".format(dataset), 'wb') as f:
    pkl.dump(tx, f)

with open("../data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("../data/ind.{}.allx".format(dataset), 'wb') as f:
    pkl.dump(allx, f)

with open("../data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)

with open("../data/ind.{}.adj".format(dataset), 'wb') as f:
    pkl.dump(adj, f)

print("Done")






