import numpy as np
from .utils import nested_dict, flatten_sample

# always picking identity for convenience
def get_sample(l):
    sample = {
        'p': [0]*l,
    }
    return nested_dict(2), sample

def unflatten_sample(flat, l):
    sample = {
        'p': flat[:l]
    }
    return sample

def execute(x, Y_train, l):
    non_zero_indices = np.nonzero(x)[0]
    y = np.sum(Y_train[non_zero_indices], axis=0)
    y = np.where(y > 0, 1, 0)
    return unflatten_sample(y, l)

def get_dataset(l):
    T, Y = [], []
    for i in range(l):
        x, y = get_sample(l)
        x['p'][i] = 1
        y['p'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))
    
    return T, np.array(Y)