import numpy as np
from .utils import flatten_sample, nested_dict

def get_sample(l):
    sample = {
        'sum_p': [0]*l,
        'sum_q': [0]*l,
        'sum_c': [0]*l
    }
    return nested_dict(2), sample

def unflatten_sample(flat, l):
    sample = {
        'sum_p': flat[:l],
        'sum_q': flat[l:2*l],
        'sum_c': flat[2*l:3*l],
    }
    return sample

def execute(x, Y_train, l):
    non_zero_indices = np.nonzero(x)[0]
    y = np.sum(Y_train[non_zero_indices], axis=0)
    y = np.where(y > 0, 1, 0)
    return unflatten_sample(y, l)


def get_dataset(l):
    T, Y = [], []
    # addition samples
    for i in range(l):
        x, y = get_sample(l)
        x['sum_p'][i] = 0
        x['sum_q'][i] = 1
        y['sum_p'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l)
        x['sum_q'][i] = 0
        x['sum_p'][i] = 1
        y['sum_p'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l)
        x['sum_p'][i] = 1
        x['sum_q'][i] = 1
        y['sum_c'][i] = 1
        y['sum_p'][i] = 0
        T.append(x)
        Y.append(flatten_sample(y))

        x, y = get_sample(l)
        x['sum_c'][i] = 1
        if i < l - 1:
            y['sum_q'][i+1] = 1
        else:
            y['sum_c'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))
    
    return T, np.array(Y)