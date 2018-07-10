#!/usr/bin/env python2

import numpy as np
import pickle
from itertools import chain
import datetime

import crf


def add_to_model(model, X, Y, offset, labels, num_features):
    height = len(X)
    width = len(X[0])
    offset_new = 0

    # helper
    y = []
    if height < 100 and width < 100:
        for i in range(height):
            for j in range(width):
                if not X[i][j] is None:
                    # gen feature vector
                    featureVector = []
                    for label in range(labels):
                        if label == Y[i][j]:
                            featureVector.append(X[i][j])
                        else:
                            e = [0]*num_features
                            featureVector.append(e)

                            f = list(chain(*featureVector))
                            # print(len(f))
                            # print(f)
                            gm.addX(offset + offset_new, np.array(X[i][j]))
                            gm.addY(offset + offset_new, Y[i][j])
                            gm.addUnary(offset + offset_new,
                                        offset + offset_new)
                            y.append(Y[i][j])

                            offset_new += 1

        print("ADDED Grid ({}x{})".format(width, height))

    else:
        print("DROPPED Grid ({}x{})".format(width, height))

        #print("Added {} Cells".format(offset_new))
    #print("Labels: {}".format(set(y)))

    return offset + offset_new


def grid_to_xy(grid, feature_indices):
    X = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]
    Y = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not grid[i][j] is None:
                X[i][j] = [grid[i][j]['X'][e] for e in feature_indices]
                Y[i][j] = grid[i][j]['Y']

    return X, Y


# Load DataSet
corpus = pickle.load(open("../data/data_train.pkl", "rb"))

trainset = list(chain.from_iterable(map(lambda x: x[1], corpus)))


num_labels = 5
num_features = 71

feature_indices = list(xrange(num_features))

offset = 0

# Create Model
gm = crf.GraphicalModel(len(feature_indices), num_labels)

Y_label = []
for idx in range(len(trainset[:50])):
    X, Y = grid_to_xy(trainset[idx], feature_indices)
    offset = add_to_model(gm, X, Y, offset, num_labels, num_features)
    Y_label = Y

print(len(trainset))

gm.print_info()

gm.learnModel()

gm.infer()


for y in range(len(Y_label)):
    print("y*{} = {}".format(y, Y_label[y]))


# Store Parameter
params = gm.getParams()
print(params)
curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
pickle.dump(
    params, open("/home/bilboi/programming/crf/data/params/unary_params{}.pickle".format(curr_time), "w"))
# Add unaries


# learn model => get_parameter

# for x in testset
#   create model
#   infer in model

# evaluate
