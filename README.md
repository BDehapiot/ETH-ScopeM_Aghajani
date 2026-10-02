![Python Badge](https://img.shields.io/badge/Python-3.10-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![TensorFlow Badge](https://img.shields.io/badge/TensoFlow-2.10-rgb(255%2C115%2C0)?logo=TensorFlow&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![CUDA Badge](https://img.shields.io/badge/CUDA-11.2-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![cuDNN Badge](https://img.shields.io/badge/cuDNN-8.1-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))    
![Author Badge](https://img.shields.io/badge/Author-Benoit%20Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2026--09--29-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))    

# ETH-ScopeM_Aghajani  
Lysosomal co-localization assay

## Index
- [Installation](#installation)
- [Method](#method)
- [Content](#content)
- [Parameters](#parameters)
- [Outputs](#outputs)
- [Comments](#comments)

## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run one of the following command:  
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf-gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf-nogpu.yml
```  
- Activate Conda environment:
```bash
conda activate Aghajani
```
Your prompt should now start with `(Aghajani)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run one of the following command: 
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf-gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf-nogpu.yml
```  
- Activate Conda environment:  
```bash
conda activate Aghajani
```
Your prompt should now start with `(Aghajani)` instead of `(base)`

</details>


## Method

<p align="left">
  <img src="utils/viewer.jpg" alt="viewer" width="768" />
</p>

We developed a Python-based (3.10) deep learning procedure to segment cellular compartments and quantify the intracellular co-localization of [*add targeted compartments description*]. Live-cell 3D Z-stacks were acquired at a single timepoint using a [*add microscope type*] equipped with an [*add objective lens type*] after incubating cells with [*add NAD+ probe & Lysotracker description*]. 

Individual channel images were extracted from a matching focal plane. To standardize signal intensities prior to model training, individual images were first normalized by dividing pixel values by the baseline intensity (10th percentile intensity), followed by pooling and rescaling overall image intensities to a [0,1] range using an outlier-robust normalization (clipping between 0.01th and 99.99th percentiles). Random patches were then cropped and manually annotated to create ground-truth training data across three categories: cytoplasm (delineated via broad LysoTracker signal), lysosomes (punctate LysoTracker structures), and NAD+-enriched intracellular structures. Three separate segmentation models based on a U-Net architecture with a ResNet-18 backbone were implemented using the segmentation_models  library (https://github.com/qubvel/segmentation_models) using Keras and TensorFlow. While the cytoplasmic model was trained directly on binary masks (cytoplasm vs. background), ground-truth annotations for the organelle models (lysosomes and NAD+ structures) were converted using a Euclidean Distance Transform (EDT) to emphasize object centroids and enable instance separation during segmentation. For model inference, cytoplasmic boundaries were defined by applying a single probability threshold (P > 0.5). Conversely, organelle masks were generated using a dual-threshold seeded watershed approach, where a lower threshold (P_low > 0.05) defined outer morphological boundaries and a higher threshold (P_high > 0.25) isolated object cores to serve as watershed seeds for separating touching objects. Organelle instances residing outside the defined cytoplasmic mask were excluded to minimize non-specific background signals. 

Co-localization was quantified at the single-object level by measuring the proportion of NAD+-conjugated probe signal overlapping with individual lysosomal masks, and these values were averaged per image. Image-level measurements were grouped across experimental conditions [*add conditions description*], and statistical significance was evaluated using [*add statistics description*]. All code and training images are publicly available on GitHub (https://github.com/BDehapiot/ETH-ScopeM_Aghajani).

## Content

- Main execution script and associated functions

```yml
- main.py
- functions.py
```

- Deep learning segmentation 

```yml
# Training scripts

- train_c1b.py 
- train_c1b.py 
- train_c1b.py 

# Model folders (weights & training info)

- .../model-sm_cyt...
- .../model-sm_c1b...
- .../model-sm_c2b...

# Annotated data folder

- .../data/train_cyt
- .../data/train_c1b
- .../data/train_c2b

# Custom dependencies folder

- .../bdtools
```

- Environment files

```yml
# GPU support

- environment_tf-gpu.yml  

# no GPU support  

- environment_tf-nogpu.yml
```

## Parameters

- procedure
```yml
# Run the a sub-section of the code
# 0 > do not run
# 1 > run only if missing
# 2 > force run 

- run_extract : int 
- run_predict : int 
- run_get_mask : int 
- run_get_results : int 
- run_display : int 
- run_plot : int 
```


- extract()
```yml
# Slice to extract

- slice_idx : int 
```

- predict()
```yml
# Rescaling factors for model training
# = 1 no rescaling
# < 1 downscale
    
- cyt_rf : float 
- c1b_rf : float
- c2b_rf : float 
```
- get_mask()
```yml
# Thresholds for binary mask creation + min. object size
# thresh0 : object (semantic segmentation)
# thresh1 : object core (instance segmentation)
# use thresh1 = None for semantic segmentation

- cyt_thresh_0 : float 
- cyt_thresh_1 : float 
- cyt_min_size : int 
    
- c1b_thresh_0 : float  
- c1b_thresh_1 : float 
- c1b_min_size : int

- c2b_thresh_0 : float  
- c2b_thresh_1 : float 
- c2b_min_size : int
```
- display()
```yml
# Napari display options 
# See https://napari.org/stable/api/napari.layers.Image.html

- C1s_gamma : float 
- C2s_gamma : float 
    
- cyt_color : str 
- c1b_color : str 
- c2b_color : str 

- cyt_opacity : float 
- c1b_opacity : float 
- c2b_opacity : float 
```

## Outputs

- Images
```yml
# Normalized images 
# From chn1 or chn2 at selected slice (slice_idx)

- C1s : uint8 
- C2s : uint8

# Prediction probabilities

- c1b_prd : uint8 
- c2b_prd : uint8  

# Binary masks

- c1b_msk : uint8  
- c1b_msk : uint8  
```

- CSV
```yml
# Image averaged results

- result_img_avg.csv 

# Condition averaged results (from image results)

- result_cnd_avg.csv  

# Columns legend

- stem            : nd2 file name
- dmf             : + or - DMF 
- chl             : + or - chloroquine
- srm             : + or - saramesine
- time            : incubation time 
- numb            : image number

- c1b_area_avg    : avg. chn1 object area (pixels)
- c1b_overlap_avg : avg. chn1 object overlap with chn2 objects
- c1b_count       : number of chn1 objects
- c1b_coverage    : total chn1 area / cytoplasmic area 
- c1b_density     : number of chn1 objects / cytoplasmic area 

- c2b_area_avg    : avg. chn2 object area (pixels)
- c2b_overlap_avg : avg. chn2 object overlap with chn1 objects
- c2b_count       : number of chn2 objects
- c2b_coverage    : total chn2 area / cytoplasmic area 
- c2b_density     : number of chn2 objects / cytoplasmic area 
```

- Misc
```yml
# Plot example
# c2b_msk overlap with c1b_msk

- result_plot.png

# Parsed file name metadata

- metadata.pkl
```



## Comments