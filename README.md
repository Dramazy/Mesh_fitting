# Bee Mesh
Surface motion capture for Bumblebees


<img width="2000" height="764" alt="1_Framework" src="https://github.com/user-attachments/assets/f2e0973d-1d2e-40e2-aeaa-6eeb179f56ed" />



##  Installation
### Create a conda environment:
~~~bash
conda create -n mesh python=3.9
conda activate mesh
~~~
### Install [Pytorch](https://pytorch.org/):
Install Pytorch based on your hardware setup. This code has been tested with pytorch==2.0.1:
~~~bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
~~~
### Install [Pytorch3d](https://github.com/facebookresearch/pytorch3d):
Follow the official [installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) carefully, as PyTorch3D requires specific builds depending on your environment.
### Install Python dependencies:
~~~bash
pip install plyfile tqdm opencv-python numpy==1.26.4 open3d einops lpips
~~~


## Dataset
Organize the dataset like:
```
<root>
|---rgbs
    |---view1
        |---<image_0001.jpg>
        |---<image_0002.jpg>
        |---<...>
    |---view2
        |---<image_0001.jpg>
        |---<image_0002.jpg>
        |---<...>

|---masks
    |---view1
        |---<image_0001.jpg>
        |---<image_0002.jpg>
        |---<...>
    |---view2
        |---<image_0001.jpg>
        |---<image_0002.jpg>
        |---<...>

|---kpts2d.npy # (num_frames, num_views, num_kpts, 2)
|---kpts3d.npy # (num_frames, num_kpts, 3)
|---cameras.npy 
|---template.pth 
        
```



## Usage
To check the articulated mesh model, see [template.ipynb](./template.ipynb) 

To peform bumblebee mesh fitting on example dataset, see [train_mesh.ipynb](./train_mesh.ipynb)

## Citing BioMTBee
If you use BioMTBee in your research, please use the following BibTeX entry.

'''
@InProceedings{10.1007/978-3-032-37432-5_34,
author="Zhang, Yi and Ye, Minchen and Zheng, Nenggan",
title="BioMTBee: Biologically Constrained Multi-view Template-Based 3D Reconstruction of Bumblebee",
booktitle="Computer Vision -- ECCV 2026",
year="2026",
pages="600--619",
}
'''

