#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:12:49 2020

@author: operator
"""

# Import libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
np.random.seed(100)
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer

# Get data
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle = True, **k)
(x, y), (xtest, ytest) = reuters.load_data(num_words = 10000)
np.load = np_load_old

# Process text
tokenizer = Tokenizer(num_words = 10000)
xtrain = tokenizer.sequences_to_matrix(x, mode = 'binary')
xtest = tokenizer.sequences_to_matrix(xtest, mode = 'binary')

ytrain = to_categorical(y)
ytest = to_categorical(ytest)

# Initialize model
model = Sequential()
model.add(Dense(512, activation = 'relu')) 
model.add(Dropout(.5))
model.add(Dense(ytrain.shape[1], activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Fit model
model.fit(xtrain, ytrain, validation_data = (xtest, ytest), epochs = 5, batch_size = 32)

# Evaluate
scores = model.evaluate(xtest, ytest, verbose = 0)
print("Accuracy: %.2f%%" % (scores[1] * 100))