import numpy as np
from .utils import flatten_sample, nested_dict

def get_sample(l):
    sample = {
        'multiplier': [0]*l,
        'to_shift_right': [0]*l,
        'multiplicand': [0]*2*l,
        'to_shift_left': [0]*2*l,
        'to_copy_to_sum_q': [0]*2*l,
        'sum_p': [0]*2*l,
        'sum_q': [0]*2*l,
        'sum_c': [0]*2*l,
        'sum_counter': [0]*4*l,
        'to_check_lsb': [0],
    }
    return nested_dict(2), sample

def unflatten_sample(flat, l):
    sample = {
        'multiplier': flat[:l],
        'to_shift_right': flat[l:2*l],
        'multiplicand': flat[2*l:4*l],
        'to_shift_left': flat[4*l:6*l],
        'to_copy_to_sum_q': flat[6*l:8*l],
        'sum_p': flat[8*l:10*l],
        'sum_q': flat[10*l:12*l],
        'sum_c': flat[12*l:14*l],
        'sum_counter': flat[14*l:18*l],
        'to_check_lsb': flat[18*l:18*l+1],
    }
    return sample

def execute(x, Y_train, l):
    non_zero_indices = np.nonzero(x)[0]
    y = np.sum(Y_train[non_zero_indices], axis=0)
    y = np.where(y > 0, 1, 0)
    return unflatten_sample(y, l)

def get_dataset(l):
    T, Y = [], []
    # preserve samples
    for i in range(l):
        # multiplier
        x, y = get_sample(l)
        x['multiplier'][i] = 1
        x['to_shift_right'][i] = 0
        if i == 0:
            x['to_check_lsb'][0] = 0
        y['multiplier'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(2*l):
        # multiplicand
        x, y = get_sample(l)
        x['multiplicand'][i] = 1
        x['to_shift_left'][i] = 0
        y['multiplicand'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    # check lsb
    x, y = get_sample(l)
    x['to_check_lsb'][0] = 1
    x['multiplier'][0] = 1
    y['multiplier'][0] = 1
    y['to_copy_to_sum_q'] = [1]*len(y['to_copy_to_sum_q'])
    T.append(x)
    Y.append(flatten_sample(y))

    x, y = get_sample(l)
    x['to_check_lsb'][0] = 1
    x['multiplier'][0] = 0
    y['to_shift_right'] = [1]*len(y['to_shift_right'])
    y['to_shift_left'] = [1]*len(y['to_shift_left'])
    T.append(x)
    Y.append(flatten_sample(y))

    # copy to addition block
    x, y = get_sample(l)
    x['multiplicand'][0] = 0
    x['to_copy_to_sum_q'][0] = 1
    y['sum_counter'][0] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    for i in range(2*l):
        x, y = get_sample(l)
        x['multiplicand'][i] = 1
        x['to_copy_to_sum_q'][i] = 1
        y['multiplicand'][i] = 1
        y['sum_q'][i] = 1
        if i == 0:
            y['sum_counter'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    # addition samples
    for i in range(2*l):
        x, y = get_sample(l)
        x['sum_p'][i] = 0
        x['sum_q'][i] = 1
        y['sum_p'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(2*l):
        x, y = get_sample(l)
        x['sum_q'][i] = 0
        x['sum_p'][i] = 1
        y['sum_p'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(2*l):
        x, y = get_sample(l)
        x['sum_p'][i] = 1
        x['sum_q'][i] = 1
        y['sum_c'][i] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(2*l-1):
        x, y = get_sample(l)
        x['sum_c'][i] = 1
        y['sum_q'][i+1] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(4*l-1):
        x, y = get_sample(l)
        x['sum_counter'][i] = 1
        y['sum_counter'][i+1] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    x, y = get_sample(l)
    x['sum_counter'][4*l-1] = 1
    y['to_shift_right'] = [1]*len(y['to_shift_right'])
    y['to_shift_left'] = [1]*len(y['to_shift_left'])
    T.append(x)
    Y.append(flatten_sample(y))

    x, y = get_sample(l)
    x['multiplier'][0] = 0
    x['to_check_lsb'][0] = 0
    x['to_shift_right'][0] = 1
    y['to_check_lsb'][0] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    x, y = get_sample(l)
    x['multiplier'][0] = 1
    x['to_check_lsb'][0] = 0
    x['to_shift_right'][0] = 1
    y['to_check_lsb'][0] = 1
    T.append(x)
    Y.append(flatten_sample(y))

    for i in range(1, l):
        x, y = get_sample(l)
        x['multiplier'][i] = 1
        x['to_shift_right'][i] = 1
        y['multiplier'][i-1] = 1
        T.append(x)
        Y.append(flatten_sample(y))

    for i in range(2*l-1):
        x, y = get_sample(l)
        x['multiplicand'][i] = 1
        x['to_shift_left'][i] = 1
        y['multiplicand'][i+1] = 1
        T.append(x)
        Y.append(flatten_sample(y))
    
    return T, np.array(Y)