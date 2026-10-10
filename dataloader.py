import os
import cv2
import numpy as np
import re
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor, Grayscale, Resize, InterpolationMode, Compose, Lambda


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


class MultiViewDataset(Dataset):
    def __init__(self, root_path, resize=None, mask_invert=False, world_scale=None):
        self.root_path = root_path
        self.rgb_paths = [] # (num_views, num_frames, 2)
        self.mask_paths = []
        self.original_sizes = [] # (num_views, num_frames, 2)
        self.kpts2d = np.load(os.path.join(root_path, 'kpts2d.npy'))[..., :2] # (num_views, num_frames, num_kpts, 2)
        self.kpts2d_normalized = None
        self.kpts3d = np.load(os.path.join(root_path, 'kpts3d.npy')) # (num_frames, num_kpts, 3)
        self.resize = resize # (h, w)
        self.mask_invert = mask_invert
        self.world_scale = world_scale
        self.rgb_tform = None
        self.mask_tform = None
 

        rgb_path = os.path.join(root_path, 'rgbs')
        mask_path = os.path.join(root_path, 'masks')
        rgb_views = sorted_alphanumeric(os.listdir(rgb_path)) # folder names in root/rgbs
        mask_views = sorted_alphanumeric(os.listdir(mask_path))
        self.num_views = len(rgb_views)
        self.num_frames = len(self.kpts3d)

        # store all the image names 
        for i in range(self.num_views):
            this_view = os.path.join(rgb_path, rgb_views[i]) # root/rgbs/view
            fnames_this_view = sorted_alphanumeric(os.listdir(this_view))
            rgb_paths = [os.path.join(this_view, fname) for fname in fnames_this_view]
            self.rgb_paths.append(rgb_paths)

            this_view = os.path.join(mask_path, mask_views[i]) # root/masks/view
            fnames_this_view = sorted_alphanumeric(os.listdir(this_view))
            mask_paths = [os.path.join(this_view, fname) for fname in fnames_this_view]
            self.mask_paths.append(mask_paths)

        # store all the image sizes (for keypoint normalization)
        for folder in self.mask_paths:
            sizes = []
            for path in folder:
                sizes.append(cv2.imread(path).shape[:2])
            self.original_sizes.append(sizes)
        self.original_sizes = np.array(self.original_sizes)

        # normalize 2d keypoints 
        self.kpts2d_normalized = self.kpts2d / np.flip(self.original_sizes, -1)[..., None, :]


        # transformation 
        rgb_tforms = [ToTensor()]
        mask_tforms = [Grayscale(num_output_channels=1), ToTensor()]
        if resize is not None:
            rgb_tforms.insert(0, Resize(self.resize, interpolation=InterpolationMode.BILINEAR))
            mask_tforms.insert(1, Resize(self.resize, interpolation=InterpolationMode.NEAREST))

        if mask_invert:
            mask_tforms.insert(1, Lambda(lambda x: torchvision.transforms.functional.invert(x)))

        self.rgb_tform = Compose(rgb_tforms)
        self.mask_tform = Compose(mask_tforms)

    
    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        # load rgbs
        rgbs = []
        for i in range(self.num_views):
            rgb_path = self.rgb_paths[i][index]
            img = default_loader(rgb_path)
            img = self.rgb_tform(img)
            rgbs.append(img)
        rgbs = torch.stack(rgbs)

        masks = []
        for i in range(self.num_views):
            mask_path = self.mask_paths[i][index]
            mask = default_loader(mask_path)
            mask = self.mask_tform(mask)
            masks.append(mask)
        masks = torch.stack(masks)

        if self.resize is not None:
            kpts2d = self.kpts2d_normalized[:, index] * np.flip(np.array(self.resize))
        else:
            kpts2d = self.kpts2d[:, index]
        kpts2d = torch.FloatTensor(kpts2d)

        kpts3d = self.kpts3d[index] * self.world_scale if self.world_scale is not None else self.kpts3d[index]
        kpts3d = torch.FloatTensor(kpts3d)

        return rgbs, masks, kpts2d, kpts3d
    

def get_loader(dataset, batch_size, shuffle=False, num_workers=4):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader



