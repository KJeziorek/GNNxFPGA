import numpy as np

def normalise(events, norm_value=256):
    '''Normalise events to range 0-255.'''
    
    x = (events['x']*norm_value/240)
    y = (events['y']*norm_value/180)
    t = events['t']
    p = events['p']
    
    t = t / 0.05
    t = (t * norm_value)
    
    events = np.column_stack((x, y, t, p))
    return events.astype(np.int32)