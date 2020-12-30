import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn 
import os

def read_folder(partition, data_dir):
    shards = []
    for fn in os.listdir(os.path.join(data_dir, partition)):
        with open(os.path.join(data_dir, partition, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    return pd.concat(shards)



def get_df(path):
    test = read_folder('test',path)
    dev = read_folder('dev',path)
    train = read_folder('train',path)
    #ram is full so we reduce 
    return test,dev,train
