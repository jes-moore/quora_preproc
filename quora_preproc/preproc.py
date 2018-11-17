import os
import gc
import time
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_and_prec(maxlen=300):
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ## split to train and val
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=2018)

    ## create training, validation, and test sets
    print('Creating train, val, and test sets')
    train_X = train_df["question_text"].values
    val_X = val_df["question_text"].values   
    test_X = test_df["question_text"].values

    ## Tokenize the sentences
    print('Tokenizing the sentences')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train_X)) 
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    print('Padding Sequences for input datasets')
    train_X = pad_sequences(train_X, maxlen=maxlen, padding='pre')
    val_X = pad_sequences(val_X, maxlen=maxlen, padding='pre')
    test_X = pad_sequences(test_X, maxlen=maxlen, padding='pre')

    ## Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values  
    
    ## Shuffling the data
    print('Shuffling training and validation sets')
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))
    val_idx = np.random.permutation(len(val_X))
    train_X = train_X[trn_idx]
    val_X = val_X[val_idx]
    train_y = train_y[trn_idx]
    val_y = val_y[val_idx]    
    
    print('Complete - Returning Values')
    return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index


def load_all_embeddings(vocabulary,max_features,embed_size):
    
    def load_embedding(embed_file):
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        if 'glove' in embed_file:
                embed_file = 'data/embeddings/glove.840B.300d/glove.840B.300d.txt'
                embeddings_index = dict(get_coefs(*o.split(" ")) for o\
                                        in open(embed_file))
        if 'wiki' in embed_file:
                embed_file = 'data/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
                embeddings_index = dict(get_coefs(*o.split(" ")) for o\
                                        in open(embed_file) if len(o)>100)
        if 'paragram' in embed_file:
                embed_file = 'data/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
                embeddings_index = dict(get_coefs(*o.split(" ")) for o\
                                        in open(embed_file, encoding="utf8", errors='ignore')\
                                        if len(o)>100)
        if 'Google' in embed_file:
                from gensim.models import KeyedVectors
                embed_file = 'data/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
                embeddings_index = KeyedVectors.load_word2vec_format(embed_file, binary=True)
                nb_words = min(max_features, len(vocabulary))
                embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
                for word, i in vocabulary.items():
                    if i >= max_features: continue
                    if word in embeddings_index:
                        embedding_vector = embeddings_index.get_vector(word)
                        embedding_matrix[i] = embedding_vector
                
                del embeddings_index; gc.collect()
                return embedding_matrix

        ## Read Embeddings for Specified File       
        all_embs = np.stack(embeddings_index.values())
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        nb_words = min(max_features, len(vocabulary))

        ## Initialize embedding matrix with random values based on -
        ## the mean and std (Not all words are contained in every pre-train)
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in vocabulary.items():
            if i >= max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
        del embeddings_index; gc.collect()
        return embedding_matrix
    
    #Loop over embeddings and create output file
    embed_list = os.listdir('data/embeddings/')
    embedding_matrices = []
    for embed in embed_list:
        embedding_matrices.append(load_embedding(embed))
        print('Loaded %s' % embed)
    #Average embedding matrice
    average_embedding_matrix = np.mean(embedding_matrices,axis=0)
    print(np.shape(average_embedding_matrix))
    
    return average_embedding_matrix