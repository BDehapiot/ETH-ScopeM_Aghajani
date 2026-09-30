#%% Imports ===================================================================

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path
from main import Main, main_parameters

# bdtools
from bdtools.model.model import Model
from bdtools.annotate import Annotate
from bdtools.patch import extract_patches

# skimage
from skimage.transform import rescale

#%% Inputs ====================================================================

parameters = {
    
    # Procedure ---------------------------------------------------------------
    
    "run_patches"  : 0,
    "run_annotate" : 0,
    "run_train"    : 1,
    
    # Paths -------------------------------------------------------------------
    
    "train_path" : Path.cwd() / "data" / "train_cyt", 
    
    # Extract -----------------------------------------------------------------
    
    "n_patches"             : 100,
    "extract_patch_size"    : 512,
    "extract_patch_overlap" : 256,
        
    }

model_parameters = {

    # Paths -------------------------------------------------------------------
    
    "root_path"          : None,
    "model_name"         : None,

    # Build -------------------------------------------------------------------
    
    "model_type"         : "sm",
    "input_shape"        : (None, None, 1),
    "loss"               : "bce",
    "metric"             : "mae",
    
    # sm
    "backbone"           : "resnet18",
    
    # # cls
    # "filters"            : [16, 32],
    # "n_classes"          : 12,
    # "classes"            : None,
    # "glob_avg_pool"      : True,
    # "regularizer"        : 0.001,
    # "dropout"            : 0.5,
    
    # # aec
    # "filters"            : [16, 32],
    # "latent_size"        : 128,
    
    # Prepare -----------------------------------------------------------------
    
    "patch_size"         : int(512 * main_parameters["cyt_rf"]),
    "patch_overlap"      : int(256 * main_parameters["cyt_rf"]),
    "mask_method"        : "binary",

    # Train -------------------------------------------------------------------
    
    "display"            : 0,
    "epochs"             : 256,
    "batch_size"         : 16,
    "validation_split"   : 0.2,
    "learning_rate"      : 0.001,
    "patience"           : 64,

    # Augment -----------------------------------------------------------------
    
    "augment_iterations" : 512,
    "augment_gamma_p"    : 0.0,
    "augment_gblur_p"    : 0.0,
    "augment_noise_p"    : 0.0,
    "augment_flip_p"     : 0.5,
    "augment_distort_p"  : 0.5,
    "augment_balance"    : False, # (cls)
    
    "augment_parameters" : {
        
    # Gamma
    "gamma_low"          : 0.75,
    "gamma_high"         : 1.25,
    "gamma_chn"          : "independent",
    
    # Gaussian blur
    "gblur_sigma_low"    : 1,
    "gblur_sigma_high"   : 3,
    "gblur_chn"          : "shared",
    
    # Noise
    "noise_gain_low"     : 30,
    "noise_gain_high"    : 60,
    "noise_std_low"      : 3,
    "noise_std_high"     : 6,
    "noise_chn"          : "independent",
    
    # Grid distort
    "distort_steps_low"  : 1,
    "distort_steps_high" : 10,
    "distort_limit_low"  : 0.1,
    "distort_limit_high" : 0.5,
        
    },

    }

#%% Class(TrainCYT) ===========================================================

class TrainCYT:
    def __init__(self, main, parameters=None, model_parameters=None):
        self.main = main
        self.parameters = parameters
        self.model_parameters = model_parameters
        self.parameters = parameters | main.parameters
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Run
        if self.run_patches:
            self.extract_patches()
        if self.run_annotate:
            self.annotate()
        if self.run_train:
            self.train()

#%% Class(TrainCYT) extract_patches() =========================================

    def extract_patches(self):
        
        # Load ----------------------------------------------------------------
        
        C1s = io.imread(self.main.C1s_path).astype("float32") / 255
        
        # Extract patches -----------------------------------------------------
        
        patches = extract_patches(
            C1s, self.extract_patch_size, self.extract_patch_overlap) 
        
        # Save ----------------------------------------------------------------
        
        idxs = np.random.choice(
            np.arange(0, len(patches)), size=self.n_patches, replace=False)
        for idx in idxs:
            patch_name = (
                f"patch_cyt_{idx:04d}.tif"
                )
            io.imsave(
                self.train_path / patch_name,
                (patches[idx] * 255).astype("uint8"), 
                check_contrast=False,
                ) 
                
#%% Class(TrainCYT) annotate() ================================================

    def annotate(self):
        Annotate(self.train_path)  
        
#%% Class(TrainCYT) train() ===================================================

    def train(self):
        
        # Load data
        msk_paths = list(self.train_path.glob("*_mask.tif"))
        msk_paths += list((self.train_path / "stock_in").glob("*_mask.tif"))
        msks, imgs = [], []
        for path in msk_paths:
            msks.append(io.imread(path))
            imgs.append(io.imread(
                str(path).replace("_mask", "")
                ))
        msks = np.stack(msks)
        imgs = np.stack(imgs)
            
        # Normalize data
        imgs = imgs.astype("float32") / 255
        
        # Rescale data
        msks = rescale(msks, (1, self.cyt_rf, self.cyt_rf), order=0)
        imgs = rescale(imgs, (1, self.cyt_rf, self.cyt_rf), order=1)
                    
        # Setup model
        self.model = Model(parameters=self.model_parameters, model_path=None)
        
        # Train model
        self.model.train(imgs, y=msks)

#%% Execute ===================================================================

if __name__ == "__main__":
    main  = Main(parameters=main_parameters)
    train = TrainCYT(
        main, parameters=parameters, model_parameters=model_parameters)