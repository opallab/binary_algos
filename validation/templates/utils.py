import numpy as np
from collections import defaultdict

def nested_dict(levels):
    if levels == 1:
        return defaultdict(dict)
    return defaultdict(lambda: nested_dict(levels - 1))

def flatten_sample(sample):
    flat = []
    for k, v in sample.items():
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)
    return flat

def set_sample(x, template):
    for k1 in template:
        for k2 in template[k1]:
            x[k1][k2] = template[k1][k2]
    return x

def match_sample(xhat, T):
    matches = []
    for i, template in enumerate(T):
        match = True 
        for k1 in template:
            for k2 in template[k1]:
                if template[k1][k2] != xhat[k1][k2]:
                    match = False
                    break
            if not match:
                break
        if match:
            matches.append(i)
    return matches

def encode_data(xhat, T):
    x = np.zeros(len(T))
    matches = match_sample(xhat, T)
    x[matches] = 1
    if x.sum() > 0:
        x = x/np.sqrt(x.sum())
    return x
