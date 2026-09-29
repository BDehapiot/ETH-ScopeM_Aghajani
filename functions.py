#%% Imports ===================================================================

import nd2
import numpy as np

# bdtools
from bdtools.norm import norm_pct

#%% Function(s) : extract =====================================================

def parse_metadata(paths):
    
    mtd = {
        "stem" : [],
        "dmf"  : [],
        "chl"  : [],
        "srm"  : [],
        "time" : [],
        "numb" : [],
        }
    
    for path in paths:
        stem = path.stem
        splt = stem.split("_")
        dmf = True if "_DMF_" in stem else False
        chl = True if "_CQ_" in stem else False
        srm = True if "_S_" in stem else False
        mtd["stem"].append(stem)
        mtd["dmf" ].append(dmf)
        mtd["chl" ].append(chl)
        mtd["srm" ].append(srm)
        mtd["time"].append(splt[-2])
        mtd["numb"].append(splt[-1])

def load_channels(path, slice_idx=5):
    arr = nd2.imread(path)
    if arr.shape[1] == 2: ch1, ch2 = 0, 1
    if arr.shape[1] == 3: ch1, ch2 = 1, 2
    C1 = arr[slice_idx, ch1, ...]
    C2 = arr[slice_idx, ch2, ...]
    return C1, C2

def normalize_channel(lst):
    for i, img in enumerate(lst):
        lst[i] = img.astype("float32") / np.quantile(img, 0.1)
    return norm_pct(lst, pct_low=0.01, pct_high=99.99, sample_fraction=0.1) 


