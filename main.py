#%% Imports ===================================================================

import pickle
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import parse_metadata, load_channels, normalize_channel

# bdtools
from bdtools.model.model import Model

#%% Inputs ====================================================================

main_parameters = {
    
    # Procedure
    "run_extract"    : 1,
    
    # Path
    "data_path"      : Path("D:\local_Aghajani\data"),
    "model_cyt_name" : "model-sm_512_binary_108-1024",
    
    # Extract
    "slice_idx"      : 5,
    
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
        self.predict_cyt()
        
#%% Class(Main) initialize() ==================================================

    def initialize(self):
        self.nd2_paths = list(self.data_path.glob("*.nd2"))
        self.mtd_path = self.data_path.parent / "metadata.pkl"
        self.C1s_path = self.data_path.parent / "C1s.tif"
        self.C2s_path = self.data_path.parent / "C2s.tif"
        
#%% Class(Main) extract() =====================================================

    def extract(self):
        
        if ((self.run_extract > 0 and not self.mtd_path.exists())
             or self.run_extract == 2):
        
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
            
#%% Class(Main) predict_cyt() =================================================

    def predict_cyt(self):
        
        # Load
        C1s = io.imread(self.C1s_path).astype("float32") / 255
        
        # Predict
        model_cyt = Model(
            parameters=None, 
            model_path=Path.cwd() / self.model_cyt_name
            )        
        cyt_prd = model_cyt.predict(
            C1s, patch_overlap=32, batch_size=32, chunk_size=128)
        
        self.cyt_prd = cyt_prd
        
        pass
        
#%% Execute ===================================================================

if __name__ == "__main__":
    main = Main(main_parameters)
    
#%%
    
    C1s = io.imread(main.C1s_path)
    C2s = io.imread(main.C2s_path)
    cyt_prd = main.cyt_prd
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(C1s, visible=1)
    vwr.add_image(C2s, visible=0)
    vwr.add_image(cyt_prd, visible=1, colormap="yellow")
