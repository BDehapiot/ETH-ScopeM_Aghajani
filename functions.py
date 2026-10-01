#%% Imports ===================================================================

import nd2
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

# bdtools
from bdtools.conn import lbl_conn
from bdtools.norm import norm_pct
from bdtools.model.model import Model

# skimage
from skimage.segmentation import watershed
from skimage.transform import rescale, resize
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops_table

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

#%% Function(s) : Main.get_result() ===========================================

def get_result(mtd, cyt_msk, c1b_msk, c2b_msk):
    
    # Nested function(s) ------------------------------------------------------
    
    def _get_result_all(mtd, msk0, msk1):

        df_all = []
        for i, (_msk0, _msk1) in enumerate(zip(msk0, msk1)):
            
            # Measure
            prp = regionprops_table(
                label(_msk0), intensity_image=_msk1, 
                properties=("area", "intensity_mean"),
                )
            prp["overlap"] = prp.pop("intensity_mean")
            
            # Handle empty prp
            if len(prp["area"]) == 0:
                prp["area"   ] = [np.nan]
                prp["overlap"] = [np.nan]
                    
            # Format & append
            for key, val in prp.items():
                prp[key] = list(val)
            for key, val in mtd.items():
                prp[key] = [val[i]] * len(prp["area"])
            df_all.append(prp)
            
        # Merge & convert to dataframe
        df_all = pd.DataFrame(df_all)
        df_all = df_all.explode(df_all.columns.tolist())
        df_all = df_all[list(mtd.keys()) + ["area", "overlap"]]
            
        return df_all
    
    def _get_result_img_avg(df_all, cols, tag="c1b"):

        df_iavg = (
            df_all.groupby(cols, as_index=False).agg(
                area_avg=("area", "mean"),
                overlap_avg=("overlap", "mean"),
                count=("area", "count"),
                )
            )
    
        # Rename cols
        df_iavg[f"{tag}_area_avg"   ] = df_iavg.pop("area_avg")
        df_iavg[f"{tag}_overlap_avg"] = df_iavg.pop("overlap_avg")
        df_iavg[f"{tag}_count"      ] = df_iavg.pop("count")
                
        return df_iavg
    
    def _get_result_cnd_avg(df_iavg, cols):
        
        df_cavg = (
            df_iavg.groupby(cols, as_index=False).agg(
                
                c1b_area_avg=("c1b_area_avg", "mean"),
                c1b_area_std=("c1b_area_avg", "std"),
                c1b_overlap_avg=("c1b_overlap_avg", "mean"),
                c1b_overlap_std=("c1b_overlap_avg", "std"),
                c1b_coverage_avg=("c1b_coverage", "mean"),
                c1b_coverage_std=("c1b_coverage", "std"),
                c1b_density_avg=("c1b_density", "mean"),
                c1b_density_std=("c1b_density", "std"),
                c1b_count=("c1b_count", "sum"),
                
                c2b_area_avg=("c2b_area_avg", "mean"),
                c2b_area_std=("c2b_area_avg", "std"),
                c2b_overlap_avg=("c2b_overlap_avg", "mean"),
                c2b_overlap_std=("c2b_overlap_avg", "std"),
                c2b_coverage_avg=("c2b_coverage", "mean"),
                c2b_coverage_std=("c2b_coverage", "std"),
                c2b_density_avg=("c2b_density", "mean"),
                c2b_density_std=("c2b_density", "std"),
                c2b_count=("c2b_count", "sum"),
                
                )
            )
        
        return df_cavg
        
    # Execute -----------------------------------------------------------------
    
    img_cols = ["stem", "dmf", "chl", "srm", "time", "numb"]
    cnd_cols = ["dmf", "chl", "srm", "time"]
    
    # Get result
    c1b_all = _get_result_all(mtd, c1b_msk, c2b_msk)
    c2b_all = _get_result_all(mtd, c2b_msk, c1b_msk)
    
    # Get image avg. result
    cyt_area = np.sum(cyt_msk, axis=(1, 2))
    c1b_iavg = _get_result_img_avg(c1b_all, img_cols, tag="c1b")
    c1b_iavg["c1b_coverage"] = np.sum(c1b_msk, axis=(1, 2)) / cyt_area
    c1b_iavg["c1b_density" ] = c1b_iavg["c1b_count"] / cyt_area
    c2b_iavg = _get_result_img_avg(c2b_all, img_cols, tag="c2b") 
    c2b_iavg["c2b_coverage"] = np.sum(c2b_msk, axis=(1, 2)) / cyt_area
    c2b_iavg["c2b_density" ] = c2b_iavg["c2b_count"] / cyt_area
    result_img_avg = pd.merge(c1b_iavg, c2b_iavg, on=img_cols)
    
    # Get condition avg. result
    result_cnd_avg = _get_result_cnd_avg(result_img_avg, cnd_cols)
    
    return result_img_avg, result_cnd_avg


