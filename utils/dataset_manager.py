import numpy as np  # linear algebra
import pandas as pd  # data processing
import kagglehub


## MNIST POINT CLOUD ##############################################################################

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


def normalize_2Dpointcloud_coordinates(C):
    """
    Normalizes a set of 2D points so that they are centered around the origin
    and scaled to fit within the unit square [0,1] x [0,1].

    Input:
    :param C: (numpy array of shape (N,2)) A set of N 2D points.

    Output:
    :return C: (numpy array of shape (N,2)) The normalized set of points.
    """

    # Center the points by subtracting the mean of each coordinate
    C = C - C.mean(0)[np.newaxis, :]

    # Shift the points so the minimum coordinate in each axis is 0
    C -= C.min(axis=0)

    # Scale the points so the maximum coordinate in each axis is 1
    C /= C.max(axis=0)

    return C



## 3D POINT CLOUD #################################################################################

def load_pointcloud3d():
    # Download latest version
    path = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")
    return path