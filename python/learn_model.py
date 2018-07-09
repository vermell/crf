#!/usr/bin/env python2

import pickle
from itertools import chain

import crf

# Load DataSet
corpus = pickle.load(open("../data/data_train.pkl", "rb"))

trainset = list(chain.from_iterable(map(lambda x: x, corpus)))

print(len(trainset))
print(trainset[0])

feature_indices = [1,2,3]



# Create Model
gm = crf.GraphicalModel(71,5)
gm.print_info()

# Add unaries

# learn model => get_parameter

# for x in testset
#   create model
#   infer in model

# evaluate
