# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:38:46 2019

@author: Matthew Nicastro
"""
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

emb = {}

with open('glove.6B.100d.txt', encoding='utf-8') as glove100: 
    for line in glove100:
        try:
            line = line.split(' ')
            key = line[0]
            embedding = line[1:]
            emb[key] = [float(dim) for dim in embedding]
        except: 
            print(line)
            break


training = pd.read_csv('train_cleaned.csv')

def buildMatrix(sentence): 
    matrix = []
    try:
        words = sentence.split(' ')
        for word in words: 
            if word in emb.keys(): 
                matrix.append(emb[word])
        return np.array(matrix).T
    except:
        return []

not_covered = 0
new_data = {}
for ind in training.index: 
    matrix = buildMatrix(training.question_text.iloc[ind])
    if len(matrix) != 0:
        if matrix.shape[1] > 1:
            pca = PCA(n_components = 1)
            pca.fit(matrix)
            new_data[training.qid.iloc[ind]] = pca.transform(matrix).flatten()
        else: 
            new_data[training.qid.iloc[ind]] = matrix.flatten()
    else: 
        not_covered += 1
print(not_covered)
new_data = pd.DataFrame.from_dict(new_data, orient='index')
new_data.to_csv('training_pca.csv')

