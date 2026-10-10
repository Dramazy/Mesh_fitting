import torch
import torch.nn as nn

from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    Rotate,
    Translate
)

from template import Template
from skinning import lbs
from render import MultiViewRenderer

class MeshDeformer(nn.Module):
    def __init__(self, batch_size, template: Template, renderer: MultiViewRenderer, device='cuda') -> None:
        super().__init__()
        self.batch_size = batch_size
        self.template: Template = template
        self.template.extend_mesh(self.batch_size)
        self.renderer: MultiViewRenderer = renderer
        self._active_params = [] # ['scale', 'pscale', 'rigid', 'tex', 'pose']
        self.device = device

        # learnable parameters
        self.scale = nn.Parameter(torch.zeros(1, 1, device=device)) # (1, 1)
        self.pscale = nn.Parameter(torch.zeros(self.template.num_bones, 3, device=device)) # (num_bones, 3)
        self.rigid = nn.Parameter(torch.zeros(batch_size, 6, device=device)) # (batch_size, 6)
        self.tex = nn.Parameter(torch.zeros(self.template.num_verts, 3, device=device)) # (num_verts, 3)
        self.pose = nn.Parameter(torch.zeros(batch_size, self.template.num_bones, 3, device=device)) # (batch_size, num_bones, 3)
        self.param_dict = {'scale': self.scale,
                           'pscale': self.pscale,
                           'rigid': self.rigid,
                           'tex': self.tex,
                           'pose': self.pose}
        
        
    def set_active(self, active_params):
        self._active_params = active_params

    def is_active(self, param_name):
        return True if param_name in self._active_params else False
    
    def reset(self, to_reset=['scale', 'pscale', 'rigid', 'tex', 'pose']):
        with torch.no_grad():
            for pname in to_reset:
                self.param_dict[pname] *= 0.

        
    def skinning(self, meshes: Meshes):
        pose_delta = self.pose if self.is_active('pose') else torch.zeros_like(self.pose)
        pose = self.template.bind_pose + pose_delta
        pose[:, self.template.root_idx] = self.template.bind_pose[self.template.root_idx] # ignore global rotation
        pscale = torch.exp(self.pscale) if self.is_active('pscale') else None
        new_verts = lbs(verts=meshes.verts_padded(),
                        bind_kinem=self.template.bind_kinem, 
                        poses=pose, 
                        bone_scale=pscale,
                        weights=self.template.skin_weights, 
                        parents=self.template.parents)
        meshes = meshes.update_padded(new_verts)
        return meshes

    def colorize(self, meshes: Meshes):
        tex = nn.Sigmoid()(self.tex) # texture should be in range(0, 1)
        meshes.textures = TexturesVertex(verts_features=tex.expand(self.batch_size, -1, -1))
        return meshes

    def rigid_tsfm(self, meshes: Meshes):
        trans, rot = self.rigid[:, :3], self.rigid[:, 3:]
        rot_mat = axis_angle_to_matrix(rot).transpose(-1, -2)
        tsfm = Rotate(rot_mat).compose(Translate(trans))
        new_verts = tsfm.transform_points(meshes.verts_padded())
        meshes = meshes.update_padded(new_verts)
        return meshes

    def scale_tsfm(self, meshes: Meshes):
        scale = torch.exp(self.scale)
        meshes = meshes.scale_verts(scale.expand(self.batch_size, -1))
        return meshes

    def map_kpts(self, meshes: Meshes):
        kpt3d = torch.matmul(self.template.mapping_matrix, meshes.verts_padded())
        all_kpt2d = []
        for i in range(len(meshes)):
            kpt = self.renderer.cameras.transform_points_screen(kpt3d[i], image_size=self.renderer.image_size)
            all_kpt2d.append(kpt[..., :2])
        kpt2d = torch.stack(all_kpt2d)
        return kpt2d, kpt3d  # (batch_size, num_views, num_kpts, 2), (batch_size, num_kpts, 3)


    def forward(self):
        meshes = self.template.extended_mesh

        if self.is_active('pose') or self.is_active('pscale'):
            meshes = self.skinning(meshes)

        if self.is_active('tex'):
            meshes = self.colorize(meshes)

        if self.is_active('scale'):
            meshes = self.scale_tsfm(meshes)

        if self.is_active('rigid'):
            meshes = self.rigid_tsfm(meshes)

        rgbs, masks = self.renderer(meshes)
        kpts2d, kpts3d = self.map_kpts(meshes)
        
        return meshes, rgbs, masks, kpts2d, kpts3d
