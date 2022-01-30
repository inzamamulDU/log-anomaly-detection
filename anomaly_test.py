#!/usr/bin/env python
# coding: utf-8


import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
import os
import time
import numpy as np
import logging
import sompy



from multiprocessing import Pool
from itertools import product
import pandas as pd
import re
import gensim as gs
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

def _preprocess(data):
    for col in data.columns:
        if col == "message":
            data[col] = data[col].apply(_clean_message)
        else:
            data[col] = data[col].apply(to_str)

    data = data.fillna("EMPTY")
    
def _clean_message(line):
    """Remove all none alphabetical characters from message strings."""
    return "".join(
        re.findall("[a-zA-Z]+", line)
    )  # Leaving only a-z in there as numbers add to anomalousness quite a bit

def to_str(x):
    """Convert all non-str lists to string lists for Word2Vec."""
    ret = " ".join([str(y) for y in x]) if isinstance(x, list) else str(x)
    return ret


def create(words, vector_length, window_size):
    """Create new word2vec model."""
    w2vmodel = {}
    for col in words.columns:
        if col in words:
            w2vmodel[col] = gs.models.Word2Vec([list(words[col])], min_count=1, size=vector_length, 
                                     window=window_size, seed=42, workers=1, iter=550,sg=0)
        else:
            #_LOGGER.warning("Skipping key %s as it does not exist in 'words'" % col)
            pass
        
    return w2vmodel

def one_vector(new_D, w2vmodel):
    """Create a single vector from model."""
    transforms = {}
    for col in w2vmodel.keys():
        if col in new_D:
            transforms[col] = w2vmodel[col].wv[new_D[col]]

    new_data = []

    for i in range(len(transforms["message"])):
        logc = np.array(0)
        for _, c in transforms.items():
            if c.item(i):
                logc = np.append(logc, c[i])
            else:
                logc = np.append(logc, [0, 0, 0, 0, 0])
        new_data.append(logc)

    return np.array(new_data, ndmin=2)


def train(inp, map_size, iterations, parallelism):
    print('training dataset is of size {inp.shape[0]}')
    mapsize = [map_size, map_size]
    np.random.seed(42)
    som = sompy.SOMFactory.build(inp, mapsize , initialization='random')
    som.train(n_job=parallelism, train_rough_len=100,train_finetune_len=5)
    model = som.codebook.matrix.reshape([map_size, map_size, inp.shape[1]])
    
    #distances = get_anomaly_score(inp, 8, model)
    #threshold = 3*np.std(distances) + np.mean(distances)
    
    return som #,threshold


def get_anomaly_score(logs, parallelism, model): # for whole dataset 
    parameters = [[x,model] for x in logs]
    pool = Pool(parallelism)
    dist = pool.map(calculate_anomaly_score, parameters) 
    #print(parameters)
    #dist = calculate_anomaly_score(parameters)
    pool.close()
    pool.join()
    #dist = [] 
    #for log in logs:
    #    dist.append(calculate_anomaly_score(log,model))
    return dist

def calculate_anomaly_score(parameters):# for a data point
    #parameters = [[x,model] for x in logs]
    log = parameters[0]
    model = parameters[1]
    #print(log)
    
    """Compute a distance of a log entry to elements of SOM."""
    dist_smallest = np.inf
    #print(model_1.shape[1])
    for x in range(model.shape[0]):
        for y in range(model.shape[1]):
            dist = cosine(model[x][y],log)
            #print(dist)
            #dist = np.linalg.norm(model[x][y] - log)
            if (dist <= dist_smallest):
                dist_smallest = dist
    return dist_smallest


def infer(w2v, som, log, data, threshold):
    
    log =  pd.DataFrame({"message":log},index=[1])
    _preprocess(log)
    
    if log.message.iloc[0] in list(w2v['message'].wv.vocab.keys()):
        vector = w2v["message"].wv[log.message.iloc[0]]
    else:
        w2v = gs.models.Word2Vec([[log.message.iloc[0]] + list(data["message"])], 
                                 min_count=1, size=100, window=3, seed=42, workers=1, iter=550, sg=0)
        vector = w2v.wv[log.message.iloc[0]]
    
    score = get_anomaly_score([vector], 1, som)
    
    if score < threshold:
        return 0, score
    else:
        return 1, score

data_path = "data-1.json"

data = pd.read_json(data_path)


_preprocess(data)
data.head()

data.tail()


x = data.message.value_counts()
for i in x.keys():
    print(i, x[i])

w2vmodel = create(data, 100,3)


transformed_data = one_vector(data, w2vmodel)
transformed_data = transformed_data[:,1:]

type(transformed_data) #numpy.ndarray
#transformed_data.shape #(10000, 100)


#unique0, counts0 = np.unique(transformed_data, return_counts=True)
#dict(zip(unique0, counts0))

from sklearn.neighbors import LocalOutlierFactor
print(__doc__)

X=transformed_data
clf = LocalOutlierFactor(n_neighbors=300, metric='euclidean')
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_

X_scores.shape


#type(y_pred)#numpy.ndarray
unique, counts = np.unique(y_pred, return_counts=True)
dict(zip(unique, counts))

#Use PCA to extract key features and visualization 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 


scaler=preprocessing.StandardScaler().fit(X) 
X_transformed=scaler.transform(X) 

print("shape of train data before pca", X_transformed.shape)


N_components=2
Data_pca=PCA(n_components=N_components).fit(X_transformed)
X_pca=Data_pca.transform(X_transformed)

print("shape of train data after pca", X_pca.shape)

unique2, counts2 = np.unique(X_pca, return_counts=True)
dict(zip(unique2, counts2))

print("variance explained", Data_pca.explained_variance_ratio_ )



y_pred_pca = clf.fit_predict(X_pca)
X_scores_pca = clf.negative_outlier_factor_


unique_pca, counts_pca = np.unique(y_pred_pca, return_counts=True)
dict(zip(unique_pca, counts_pca))



map_size = 16
model = train(transformed_data, map_size, 0, 2)


m = model.codebook.matrix.reshape([16, 16, transformed_data.shape[1]])

#then = time.
distances = get_anomaly_score(transformed_data, 8, m)

print(max(distances))
print(min(distances))
print(np.mean(distances))
print(np.std(distances))


threshold = 3*np.std(distances) + np.mean(distances)
threshold

count = 0
for i, j in enumerate(distances):
    if j > threshold:
        print(j, data.loc[i])
        count += 1
print(count)

data.message.value_counts()


infer(w2vmodel,m,'rootCMDusrlocalbinmonitorapachestatsshdevnull', data,threshold)

infer(w2vmodel,m,'DEBUGProcessedoutofpackages', data,threshold)

infer(w2vmodel,m,'ThreadodcsserverINFOComposetypestatedoneComposedone ', data,threshold)

infer(w2vmodel,m,'FSKSDNKSDNKSfdhfhfdhfdhfdhfdhfd  hfhf dlksfljscomLDN', data,threshold)

infer(w2vmodel,m,'this is not a log message',data,threshold)

infer(w2vmodel,m,'message',data,threshold)




