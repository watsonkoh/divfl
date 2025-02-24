import json, math, os, sys
import numpy as np
import random
from tqdm import trange


NUM_USER = 30

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate_synthetic(alpha, beta, iid):
    dimension = 60
    NUM_CLASS = 10

    #samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    samples_per_user = 50*np.ones((NUM_USER,))
    samples_per_user = np.array(samples_per_user, int)
    print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### define some eprior ####
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        mean_x[i] = np.zeros(dimension)

    W = np.random.normal(0, 1, (dimension, NUM_CLASS))
    b = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(NUM_USER):
        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))

    return X_split, y_split



def main():
    train_path = "data/train/mytrain.json"
    test_path = "data/test/mytest.json"

    X, y = generate_synthetic(alpha=0, beta=0, iid=1)

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in trange(NUM_USER, ncols=120):

        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outtrainfile:
        json.dump(train_data, outtrainfile)
    with open(test_path, 'w') as outtestfile:
        json.dump(test_data, outtestfile)


if __name__ == "__main__":
    main()

