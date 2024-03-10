import numpy as np

def normalise(events, norm_value=255):
    '''Normalise events to range 0-255.'''
    
    x = (events['x']/240)
    x = (x*norm_value).astype(np.int32)

    y = (events['y']/180)
    y = (y*norm_value).astype(np.int32)

    t = events['t'] - events['t'][0]
    t = t/t[-1]
    t = (t * norm_value).astype(np.int32)
    
    p = events['p']
    return np.column_stack((x, y, t, p))