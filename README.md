# Bee Mesh
Surface motion capture for Bumblebees

## Description
**BioMTBee** is a template-based 3D mesh reconstruction framework for bumblebees. We built a high-fidelity articulated mesh template from micro-CT scans to provide an accurate morphological prior. A multi-view 3D pose estimator with spatio-temporal filtering (BPST) is then introduced to extract robust 3D keypoints from 2D detections, guiding stable mesh template fitting. Building on this template, we combine 3D pose and shape supervision with biological constraints—bilateral symmetry,
kinematic coupling, and temporal smoothness—to achieve anatomically consistent and temporally stable surface reconstructions. Experiments on 3D reconstruction from multi-view images show improvements in morphological accuracy and biological plausibility over representative baselines, capturing fast limb movements and fine structural and textural details, and also generalize well to cross-species reconstruction.

<img width="2000" height="764" alt="1_Framework" src="https://github.com/user-attachments/assets/f2e0973d-1d2e-40e2-aeaa-6eeb179f56ed" />
Fig. 1: Method overview. We design a biologically grounded reconstruction pipeline:(1) build an articulated micro-CT bumblebee mesh template for anatomical consistency; (2) estimate robust 3D pose keypoints via BPST with HQ-SAM masks to guide deformation; (3) learn nonrigid deformations to adapt the template to individuals; (4) enforce data fidelity and biological plausibility via loss design; (5) optimize all variables through differentiable rendering.

<img width="1740" height="1096" alt="5_reconstruction_results" src="https://github.com/user-attachments/assets/9eedf822-65d8-4926-a823-b8382a038d74" />
Fig. 2: Mesh reconstruction results of a bumblebee in motion. Despite rapid limb movements, our method accurately captures nonrigid deformations and reconstructs
meshes closely aligned with the ground truth.

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

```
@InProceedings{10.1007/978-3-032-37432-5_34,
author="Zhang, Yi and Ye, Minchen and Zheng, Nenggan",
title="BioMTBee: Biologically Constrained Multi-view Template-Based 3D Reconstruction of Bumblebee",
booktitle="Computer Vision -- ECCV 2026",
year="2026",
pages="600--619",
}
```

