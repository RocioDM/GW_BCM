import numpy as np  # linear algebra
import pandas as pd  # data processing
import scipy as sp
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import time
import kagglehub
import ot



def load_pointcloudmnist2d():
    # Download the dataset from Kaggle and save the path
    path = kagglehub.dataset_download("cristiangarcia/pointcloudmnist2d")
    # Load the test dataset from the downloaded files
    df = pd.read_csv(path + "/test.csv")
    # Extract numerical data (point cloud coordinates) from the dataset, excluding the first column (which contains labels)
    Data = df[df.columns[1:]].to_numpy()
    # Extract labels (digits) from the first column
    label = df[df.columns[0]].to_numpy()
    # Reshape data into an array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
    Data = Data.reshape(Data.shape[0], -1, 3)
    # Create a list of indices for each digit (0-9), grouping their occurrences in the dataset
    digit_indices = [np.where(label == digit)[0].tolist() for digit in range(10)]
    return Data, label, digit_indices