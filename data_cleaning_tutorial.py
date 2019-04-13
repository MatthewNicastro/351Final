# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:22:32 2019

@author: Matthew Nicastro
"""
import pandas as pd
import data_cleaning as dc

emb = set()

with open('embeddings/master.txt', encoding='latin') as words: 
    for line in words:
        try:
            line = line.split(' ')
            key = line[0]
            emb.add(key)
        except: 
            print(line)
            break

train = pd.read_csv('data/not_cleaned/train.csv')
test = pd.read_csv('data/not_cleaned/test.csv')

questions = pd.concat([train.question_text, test.question_text])

questions = questions.apply(lambda x: x.lower())
vocab = dc.build_vocab(questions)
currVocab, notKnown = dc.coverage(vocab, emb)


questions = questions.apply(lambda x: dc.cleanCont(x, emb))
vocab = dc.build_vocab(questions)
currVocab, notKnown = dc.coverage(vocab, emb)

questions = questions.apply(lambda x: dc.clean_characters(x, emb))
vocab = dc.build_vocab(questions)
currVocab, notKnown = dc.coverage(vocab, emb)

dc.initworddict()
questions = questions.apply(lambda x: dc.correct_spelling(x))
vocab = dc.build_vocab(questions)
currVocab, notKnown = dc.coverage(vocab, emb)


dc.initworddict()

train.question_text = train.question_text.apply(lambda x: x.lower())
train.question_text = train.question_text.apply(lambda x: dc.cleanCont(x, emb))
train.question_text = train.question_text.apply(lambda x: dc.clean_characters(x, emb))
train.question_text = train.question_text.apply(lambda x: dc.correct_spelling(x))

test.question_text = test.question_text.apply(lambda x: x.lower())
test.question_text = test.question_text.apply(lambda x: dc.cleanCont(x, emb))
test.question_text = test.question_text.apply(lambda x: dc.clean_characters(x, emb))
test.question_text = test.question_text.apply(lambda x: dc.correct_spelling(x))

train.to_csv('train_cleaned_v4.csv', index = False)
test.to_csv('test_cleaned_v4.csv', index = False)
