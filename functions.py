#%% Imports ===================================================================

import nd2
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

# bdtools
from bdtools.conn import lbl_conn
from bdtools.norm import norm_pct
from bdtools.model.model import Model

# skimage
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.transform import rescale, resize
from skimage.morphology import remove_small_objects

#%% Function(s) : Main.extract() ==============================================

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
        
    return mtd

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

#%% Function(s) : Main.predict() ==============================================

def predict(
        arr, model_name=None, rf=0.5,
        patch_overlap=32, batch_size=32, chunk_size=128,
        ):
    
    # Initialize
    shape = arr.shape
    model = Model(parameters=None, model_path=Path.cwd() / model_name)   
    
    # Rescale
    if rf != 1:
        arr = rescale(arr, (1, rf, rf), order=1)
        
    # Predict
    prd = model.predict(
        arr, 
        patch_overlap=patch_overlap, 
        batch_size=batch_size, 
        chunk_size=chunk_size,
        )
    
    if rf != 1:
        prd = resize(prd, shape)
        
    return prd

#%% Function(s) : Main.get_mask() =============================================

def get_mask(prd, thresh_0=0.5, thresh_1=None, min_size=0):
    
    def split_objects(prd, msk_0, msk_1):
        msk = watershed(-prd, label(msk_1), mask=msk_0)
        msk[lbl_conn(msk, conn=2) == 2] = 0
        return msk > 0
    
    # Get mask
    if thresh_1 is not None:
        
        msk_0 = prd > thresh_0
        msk_1 = prd > thresh_1
    
        if prd.ndim == 3:
            msk = Parallel(n_jobs=-1)(
                delayed(split_objects)(_prd, _msk_0, _msk_1)
                for _prd, _msk_0, _msk_1 in zip(prd, msk_0, msk_1)
                )
            msk = np.stack(msk)
        
        elif prd.ndim == 2:
            msk = split_objects(prd, msk_0, msk_1)

    else:
        
        msk = prd > thresh_0
    
    # Remove small objects
    if min_size > 0:
        msk = remove_small_objects(msk, min_size=min_size)
        
    return msk


