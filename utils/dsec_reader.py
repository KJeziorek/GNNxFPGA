import numpy as np
import math
import h5py

# From Dsec: https://github.com/uzh-rpg/dsec-det/blob/master/src/dsec_det/io.py

class DSECReader:
    def __init__(self, h5_file):
        self.h5f = h5py.File(str(h5_file), 'r')
    
    def max_time(self):
        return self.h5f['events']['t'][-1]
    
    def close(self):
        self.h5f.close()

    def _extract_from_h5_by_index(self, ev_start_idx: int, ev_end_idx: int):
        events = self.h5f['events']
        x, y, t, p = events['x'], events['y'], events['t'], events['p']

        x_new = x[ev_start_idx:ev_end_idx]
        y_new = y[ev_start_idx:ev_end_idx]
        p_new = p[ev_start_idx:ev_end_idx]
        t_new = t[ev_start_idx:ev_end_idx].astype("int64") + self.h5f["t_offset"][()]

        output = {'p': p_new, 't': t_new, 'x': x_new, 'y': y_new,}
        return output

    def extract_timewindow(self, t_min_us: int, t_max_us: int):
        ms2idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')
        t_offset = self.h5f['t_offset'][()]

        events = self.h5f['events']
        t = events['t']

        t_ev_start_us = t_min_us
        assert t_ev_start_us >= t[0], (t_ev_start_us, t[0])
        t_ev_start_ms = t_ev_start_us // 1000
        ms2idx_start_idx = int(t_ev_start_ms)
        ev_start_idx = ms2idx[ms2idx_start_idx]

        t_ev_end_us = t_max_us
        assert t_ev_end_us <= t[-1], (t_ev_end_us, t[-1])
        t_ev_end_ms = math.floor(t_ev_end_us / 1000)
        ms2idx_end_idx = t_ev_end_ms
        ev_end_idx = ms2idx[ms2idx_end_idx]

        return self._extract_from_h5_by_index(ev_start_idx, ev_end_idx)