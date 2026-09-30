#%% Imports ===================================================================

import pickle
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path

# functions
from functions import (
    parse_metadata, load_channels, normalize_channel,
    predict, get_mask,
    )

#%% Inputs ====================================================================

main_parameters = {
    
    # Procedure
    "run_extract"    : 1,
    "run_predict"    : 1,
    "run_get_mask"   : 1,
    "run_analyse"    : 0,
    
    # Path
    "data_path"      : Path("D:\local_Aghajani\data"),
    "model_cyt_name" : "model-sm_cyt_256_binary_121-512",
    "model_c1b_name" : "model-sm_c1b_256_edt_134-512",
    "model_c2b_name" : "model-sm_c2b_256_edt_112-512",
    "tags"           : ["cyt", "c1b", "c2b"],
    
    # extract()
    "slice_idx"      : 5,
    
    # predict()
    "cyt_rf"         : 0.5,
    "c1b_rf"         : 0.5,
    "c2b_rf"         : 0.5,
    
    # get_mask()
    "cyt_thresh_0"   : 0.75, 
    "cyt_thresh_1"   : None,
    "cyt_min_size"   : 128,

    "c1b_thresh_0"   : 0.1, 
    "c1b_thresh_1"   : 0.5,
    "c1b_min_size"   : 16,

    "c2b_thresh_0"   : 0.1, 
    "c2b_thresh_1"   : 0.5,
    "c2b_min_size"   : 16,
    
    }

#%% Class(Main) ===============================================================

class Main:
    def __init__(self, parameters):
        self.parameters = parameters
        for key, val in self.parameters.items():
            setattr(self, key, val)
            
        # Run
        self.initialize()
        self.extract()
        self.predict()
        self.get_mask()
        self.analyse()
        
#%% Class(Main) initialize() ==================================================

    def initialize(self):
        
        # Paths
        self.root_path = self.data_path.parent
        self.nd2_paths = list(self.data_path.glob("*.nd2"))
        self.mtd_path  = self.root_path / "metadata.pkl"
        self.C1s_path  = self.root_path / "C1s.tif"
        self.C2s_path  = self.root_path / "C2s.tif"
        for tag in self.tags:
            tag_prd = f"{tag}_prd"
            tag_msk = f"{tag}_msk"
            setattr(self, tag_prd + "_path", self.root_path / (tag_prd + ".tif"))
            setattr(self, tag_msk + "_path", self.root_path / (tag_msk + ".tif"))
        
#%% Class(Main) extract() =====================================================

    def extract(self):
        
        if ((self.run_extract > 0 and not self.mtd_path.exists())
             or self.run_extract == 2):
            
            print("Main.extract()")
        
            # Metadata
            mtd = parse_metadata(self.nd2_paths)
            
            # Load
            C1s, C2s = [], []
            for nd2_path in self.nd2_paths:
                C1, C2 = load_channels(nd2_path, slice_idx=self.slice_idx)
                C1s.append(C1)
                C2s.append(C2)
                
            # Normalize & format
            C1s = normalize_channel(C1s)
            C2s = normalize_channel(C2s)
            C1s = (np.stack(C1s) * 255).astype("uint8")
            C2s = (np.stack(C2s) * 255).astype("uint8")
                    
            # Save
            with open(self.mtd_path, "wb") as f: 
                pickle.dump(mtd, f)
            io.imsave(self.C1s_path, C1s, check_contrast=False)
            io.imsave(self.C2s_path, C2s, check_contrast=False) 
            
#%% Class(Main) predict() =====================================================

    def predict(self):
        
        # Load
        C1s = io.imread(self.C1s_path).astype("float32") / 255
        C2s = io.imread(self.C2s_path).astype("float32") / 255
        
        for tag in self.tags:
            
            prd_path = getattr(self, f"{tag}_prd_path")
            
            if ((self.run_predict > 0 and not prd_path.exists())
                 or self.run_predict == 2):
                
                print(f"Main.predict() - {tag}")
            
                arr = C2s if tag == "c2b" else C1s 
                model_name = getattr(self, f"model_{tag}_name")
                rf=getattr(self, f"{tag}_rf")
                
                # Predict
                prd = predict(
                    arr, model_name=model_name, rf=rf,
                    patch_overlap=32, batch_size=32, chunk_size=None,
                    )
    
                # Save
                io.imsave(
                    getattr(self, f"{tag}_prd_path"), 
                    (prd * 255).astype("uint8"), 
                    check_contrast=False,
                    )
                        
#%% Class(Main) get_mask() ====================================================

    def get_mask(self):
        
        for tag in self.tags: 
            
            msk_path = getattr(self, f"{tag}_msk_path")
            
            if ((self.run_get_mask > 0 and not msk_path.exists())
                 or self.run_get_mask == 2):

                print(f"Main.get_mask() - {tag}")

                # Load 
                prd = io.imread(getattr(self, f"{tag}_prd_path"))
                
                # Get mask
                msk = get_mask(
                    prd.astype("float32") / 255,
                    thresh_0=getattr(self, f"{tag}_thresh_0"), 
                    thresh_1=getattr(self, f"{tag}_thresh_1"), 
                    min_size=getattr(self, f"{tag}_min_size"), 
                    )
                
                # Save
                io.imsave(
                    getattr(self, f"{tag}_msk_path"), 
                    (msk * 255).astype("uint8"), 
                    check_contrast=False,
                    )

#%% Class(Main) analyse() =====================================================

    def analyse(self):
        
        # # Load
        # cyt_msk = io.imread(self.cyt_msk_path)
        # c1b_msk = io.imread(self.c1b_msk_path)
        # c2b_msk = io.imread(self.c2b_msk_path)
        
        # # 
        # c1b_msk[~cyt_msk] = 0
        # c2b_msk[~cyt_msk] = 0
        
        pass

#%% Execute ===================================================================

if __name__ == "__main__":
    main = Main(main_parameters)
    
#%%
    
    from skimage.measure import label, regionprops_table

    # -------------------------------------------------------------------------

    C1s = io.imread(main.C1s_path)
    C2s = io.imread(main.C2s_path)

    # -------------------------------------------------------------------------

    # Load
    with open(main.mtd_path, "rb") as file:
        mtd = pickle.load(file)
    cyt_msk = io.imread(main.cyt_msk_path).astype(bool)
    c1b_msk = io.imread(main.c1b_msk_path).astype(bool)
    c2b_msk = io.imread(main.c2b_msk_path).astype(bool)
    
    #
    c1b_msk[cyt_msk == 0] = 0
    c2b_msk[cyt_msk == 0] = 0
    
    #   
    data = []
    for i, (_c1b_msk, _c2b_msk) in enumerate(zip(c1b_msk, c2b_msk)):
        
        prp = regionprops_table(
            label(_c2b_msk), intensity_image=_c1b_msk, 
            properties=(
                "centroid", 
                "area", 
                "intensity_mean"
                ),
            )
        prp["ypos"   ] = prp.pop("centroid-0")
        prp["xpos"   ] = prp.pop("centroid-1")
        prp["overlap"] = prp.pop("intensity_mean")
        
        for key, val in prp.items():
            prp[key] = list(val)
        
        for key, val in mtd.items():
            prp[key] = [val[i]] * len(prp["area"])
        
        data.append(prp)
        
    df = pd.DataFrame(data)
    df = df.explode(df.columns.tolist())
        
        # c2_prp[]
        # mtd["data"].append(c2b_prp)
        

    # -------------------------------------------------------------------------
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     C1s, visible=1,
    #     gamma=0.25,
    #     )
    # vwr.add_image(
    #     C2s, visible=0,
    #     gamma=0.25,
    #     )
    # vwr.add_image(
    #     cyt_msk, visible=1,
    #     colormap="gray", opacity=0.1,
    #     )
    # vwr.add_image(
    #     c1b_msk, visible=1,
    #     colormap="bop orange", opacity=0.25,
    #     )
    # vwr.add_image(
    #     c2b_msk, visible=1,
    #     colormap="bop blue", opacity=0.25,
    #     )
