#%% Imports ===================================================================

import pickle
import napari
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import (
    parse_metadata, load_channels, normalize_channel,
    predict, get_mask, get_result,
    )

# QT
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

#%% Inputs ====================================================================

main_parameters = {
    
    # Procedure
    "run_extract"    : 1,
    "run_predict"    : 1,
    "run_get_mask"   : 1,
    "run_get_result" : 1,
    "run_display"    : 1,
    
    # Path
    "data_path"      : Path("D:\local_Aghajani\data"),
    "model_cyt_name" : "model-sm_cyt_256_binary_121-512",
    "model_c1b_name" : "model-sm_c1b_256_edt_134-512",
    "model_c2b_name" : "model-sm_c2b_256_edt_112-512",
    "tags"           : ["cyt", "c1b", "c2b"],
    
    # extract() ---------------------------------------------------------------
    
    "slice_idx"      : 5,
    
    # predict() ---------------------------------------------------------------
    
    "cyt_rf"         : 0.5,
    "c1b_rf"         : 0.5,
    "c2b_rf"         : 0.5,
        
    # get_mask() --------------------------------------------------------------
    
    "cyt_thresh_0"   : 0.75, 
    "cyt_thresh_1"   : None,
    "cyt_min_size"   : 128,
    
    "c1b_thresh_0"   : 0.1, 
    "c1b_thresh_1"   : 0.5,
    "c1b_min_size"   : 16,

    "c2b_thresh_0"   : 0.1, 
    "c2b_thresh_1"   : 0.5,
    "c2b_min_size"   : 16,
    
    # display() ---------------------------------------------------------------
    
    "C1s_gamma"      : 0.5,
    "C2s_gamma"      : 0.5,
        
    "cyt_color"      : "gray",
    "c1b_color"      : "bop blue",
    "c2b_color"      : "bop orange",
    
    "cyt_opacity"    : 0.05,
    "c1b_opacity"    : 0.25,
    "c2b_opacity"    : 0.25,

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
        self.get_result()
        self.display()
        
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
        self.result_img_avg_path = self.root_path / "result_img_avg.csv"
        self.result_cnd_avg_path = self.root_path / "result_cnd_avg.csv"
        
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
                setattr(self, f"{tag}_msk", get_mask(
                    prd.astype("float32") / 255,
                    thresh_0=getattr(self, f"{tag}_thresh_0"), 
                    thresh_1=getattr(self, f"{tag}_thresh_1"), 
                    min_size=getattr(self, f"{tag}_min_size"), 
                    ))
                
        if ((self.run_get_mask > 0 and not msk_path.exists())
             or self.run_get_mask == 2):
                
            # Remove out of "cyt" blobs
            self.c1b_msk[self.cyt_msk == 0] = 0
            self.c2b_msk[self.cyt_msk == 0] = 0
                
            # Save
            for tag in self.tags: 
                msk = getattr(self, f"{tag}_msk")
                io.imsave(
                    getattr(self, f"{tag}_msk_path"), 
                    (msk * 255).astype("uint8"), 
                    check_contrast=False,
                    )

#%% Class(Main) get_result() ==================================================

    def get_result(self):

        if ((self.run_get_result > 0 and not self.result_img_avg_path.exists())
             or self.run_get_result == 2):
        
            print("Main.get_result()")
            
            # Load
            with open(self.mtd_path, "rb") as file:
                mtd = pickle.load(file)
            cyt_msk = io.imread(self.cyt_msk_path).astype(bool)
            c1b_msk = io.imread(self.c1b_msk_path).astype(bool)
            c2b_msk = io.imread(self.c2b_msk_path).astype(bool)
            
            # Get result
            result_img_avg, result_cnd_avg = get_result(
                mtd, cyt_msk, c1b_msk, c2b_msk)
            
            # Save
            result_img_avg.to_csv(self.result_img_avg_path, index=False)
            result_cnd_avg.to_csv(self.result_cnd_avg_path, index=False)

#%% Class(Main) display() =====================================================
        
    def display(self):
        
        if self.run_display > 0:
             
            # Init. -----------------------------------------------------------
            
            self.vwr = napari.Viewer()
            self.current_slice = 0
            
            # Load
            C1s = io.imread(self.C1s_path)
            C2s = io.imread(self.C2s_path)
            
            # Init. layers
            self.vwr.add_image(
                C1s, name="C1s", colormap=self.c1b_color,
                blending="additive", gamma=self.C1s_gamma,
                )
            self.vwr.add_image(
                C2s, name="C2s", colormap=self.c2b_color,
                blending="additive", gamma=self.C2s_gamma,
                )
            
            for tag in self.tags:
                
                tag_msk = f"{tag}_msk"
                
                # Load
                setattr(self, tag_msk, io.imread(
                    getattr(self, tag_msk + "_path")))
                
                # Init. layers
                self.vwr.add_image(
                    getattr(self, tag_msk), 
                    name=tag_msk,
                    colormap=getattr(self, f"{tag}_color"),
                    opacity=getattr(self, f"{tag}_opacity"),
                    blending="additive",
                    )
                
            # Dock ------------------------------------------------------------
            
            # Create texts
            self.info = QLabel()
            self.info.setFont(QFont("Consolas"))
            self.get_info()
            
            # Create layout
            self.layout = QVBoxLayout()
            self.layout.addWidget(self.info)

            # Create widget
            self.widget = QWidget()
            self.widget.setLayout(self.layout)
            self.vwr.window.add_dock_widget(
                self.widget, area="right", name="Painter") 
            
            # Callbacks -------------------------------------------------------
            
            def on_slider_change(event):
                self.current_slice = event.value[0]
                self.get_info()
            
            self.vwr.dims.events.current_step.connect(on_slider_change)
            
            # Shortcuts -------------------------------------------------------
                        
            @self.vwr.bind_key("Delete", overwrite=True)
            def toogle_masks_key(viewer):
                self.toogle_masks(False)
                yield
                self.toogle_masks(True)
                
    # Display function(s) -----------------------------------------------------
                
    def toogle_masks(self, visible):
        for tag in self.tags:
            self.vwr.layers[f"{tag}_msk"].visible = visible
            
    def get_info(self):
        self.info.setText(self.nd2_paths[self.current_slice].name)

#%% Execute ===================================================================

if __name__ == "__main__":
    main = Main(main_parameters)
        
#%% 

    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load
    df_img = pd.read_csv(main.result_img_avg_path)
    df_cnd = pd.read_csv(main.result_cnd_avg_path)
    
    # Initialize
    n = 6
    conditions = ["Chloroquine", "Siramesine"]
    cnd_bool = df_cnd.iloc[:n, 1:3].to_numpy()
    cnd_bool = cnd_bool[::2, :].T
    cnd_txt = np.where(cnd_bool, "+", "-")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    x = np.arange(n)
    width = 0.25
    
    for i in range(n):
        j = i // 2
        is_even = i % 2 == 0
        shift = - width / 1.8 if is_even else width / 1.8
        color = "silver" if is_even else "khaki"
        
        c2b_overlap_avg = df_cnd.at[i, "c2b_overlap_avg"]
        c2b_overlap_std = df_cnd.at[i, "c2b_overlap_std"]
        ax.bar(
            j + shift, c2b_overlap_avg, width, 
            yerr=c2b_overlap_std, capsize=4,    
            error_kw={"ecolor": "black", "elinewidth": 1, "capthick": 1},
            color=color,
            )
    
    # Condition table
    table = ax.table(
        cellText=cnd_txt,
        rowLabels=conditions,
        fontsize=12,
        loc="bottom",
        cellLoc="center",
        bbox=[0, -0.35, 1, 0.25]
        )
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)
        
    # Format
    ax.set_title("c2b/c1b overlap %")
    ax.set_xlim([-0.5, 2.5])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["0h / 1h"] * 3)
    ax.set_ylabel("overlap %", fontsize=12)
      
    plt.tight_layout()
    plt.show()    