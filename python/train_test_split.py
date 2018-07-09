#!/usr/bin/env python2

import pandas as pd
import pickle
import random
import argparse

def main(filename, output_directory, fraction = 0.7):
    corpus = pickle.load( open( filename, "rb" ) )

    print("Size Corpus {}".format(len(corpus)))
    print("Spilt {}:{}".format(int(fraction * 100), int(100 *(1 - fraction))))


    random.shuffle(corpus)
    split_at = int(len(corpus) * fraction)
    train_validation_data = corpus[:split_at]
    test_data = corpus[split_at:]

    print("Size of Test-Dataset: {}".format(len(test_data)))
    print("Size of Training-Dataset: {}".format(len(train_validation_data)))

    # Save this split
    output = open("{}/data_train.pkl".format(output_directory), 'wb')
    pickle.dump(train_validation_data, output)
    output.close()
    output = open("{}/data_test.pkl".format(output_directory), 'wb')
    pickle.dump(test_data, output)
    output.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="path to pickle file containing the corpus i.e data.pkl", type=str)
    parser.add_argument("output_directory", help="Path to output-folder.", type=str)
    parser.add_argument("-f", "--fraction", help="Fraction of datums in training-set.", type=float)
    args = parser.parse_args()

    if args.fraction:
        main(args.filename, args.output_directory, args.fraction)
    else:
        main(args.filename, args.output_directory)
        
