import numpy as np

def normalise(events, norm_value=256, x_max=240, y_max=180, t_max=0.05):
    '''Normalise events to norm_value.'''
    
    x = (events['x']*(norm_value)/x_max)
    y = (events['y']*(norm_value)/y_max)
    t = events['t']
    p = events['p']
    
    t = t / t_max
    t = (t * (norm_value))
    
    events = np.column_stack((x, y, t, p))
    return events.astype(np.int32)