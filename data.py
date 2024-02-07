import pickle
import numpy as np

def load_data():
    f = open('data/data_batch_1', 'rb')
    datadict = pickle.load(f, encoding='bytes')
    f.close()

    # Load and reshape data
    X = datadict[b'data'].reshape(-1, 3, 32, 32)
    Y = datadict[b'labels']

    X = (X / 255.0).astype(np.float32)
    mean = X.mean(axis = (0, 2, 3))
    std = X.std(axis = (0, 2, 3))
    X = (X - mean[:, None, None]) / std[:, None, None]

    # # Normalize
    # X = X.astype("float")
    # X /= 255.0

    # # Standardize
    # mean = np.mean(X, axis=(0, 2, 3))
    # std = np.std(X, axis=(0, 2, 3))
    # X = (X - mean[:, None, None]) / std[:, None, None]

    # X = np.array(X)
    Y = np.array(Y)

    return X, Y