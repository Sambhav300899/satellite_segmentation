import os
import pandas as pd
from labels import labels

if __name__ == '__main__':
    df = pd.read_csv('dataset/train_wkt_v4.csv')
    df.columns = ['ImageId', df.columns[1], df.columns[2]]
    df.to_csv('dataset/train_wkt_v4.csv')
