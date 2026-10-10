import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    AmbientLights
)


def get_raster_settings(image_size, sigma=1e-6, faces_per_pixel=10):
    raster_settings_soft = RasterizationSettings(
        image_size = image_size, 
        blur_radius = np.log(1. / 1e-4 - 1.) * sigma, 
        faces_per_pixel = faces_per_pixel,
        perspective_correct = False,
    )
    return raster_settings_soft


def get_image_render(setting, cameras, lights):
    renderer = MeshRenderer(
        rasterizer = MeshRasterizer(
            cameras = cameras, 
            raster_settings = setting
        ),
        shader = SoftPhongShader(
            device = cameras.device, 
            cameras = cameras,
            lights = lights)
    )  
    return renderer


class MultiViewRenderer(nn.Module):
    def __init__(self, image_size, cameras, lights=None, sigma=1e-6, faces_per_pixel=10, device='cpu'):
        super().__init__()
        self.num_views = len(cameras)
        self.cameras = cameras.to(device)
        self.lights = lights.to(device) if lights is not None else AmbientLights(device=device) # set to ambient light by default
        self.image_size= (image_size, image_size) if isinstance(image_size, int) else image_size # (h, w)
        self.raster_settings = get_raster_settings(self.image_size, sigma, faces_per_pixel)
        self.renderer = get_image_render(self.raster_settings, self.cameras, self.lights)

    def forward(self, meshes: Meshes):
        num_frames = len(meshes)
        all_images = []
        
        for i in range(num_frames):
            mesh = meshes[i].extend(self.num_views)
            images = self.renderer(mesh) # (v, h, w, c)
            all_images.append(images)
            
        all_images = rearrange(torch.stack(all_images), 'f v h w c -> f v c h w')
        rgbs, masks = all_images[..., :3, :, :], all_images[..., 3, None, :, :]
        return rgbs, masks # (f, v, 3, h, w), (f, v, 1, h, w)