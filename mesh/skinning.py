import copy
import torch
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix, Scale
from pytorch3d.structures import Meshes

def estimate_global_tsfm(bone: torch.Tensor):
    """
    - bone: (2, 3), bone[0] is head and bone[1] is tail
    """
    T_world2bone = torch.eye(4, 4)
    bone_vec = (bone[1] - bone[0]).reshape(1, 3)
    up = torch.tensor([0., 1, 0]).reshape(1, 3)
    vec_forward = bone_vec / torch.linalg.norm(bone_vec)
    vec_right = torch.cross(up, vec_forward)
    vec_right = vec_right / torch.linalg.norm(vec_right)
    vec_up = torch.cross(vec_forward, vec_right)
    vec_up = vec_up / torch.linalg.norm(vec_up)
    rot_mat = torch.hstack([vec_right.reshape(3, 1), vec_up.reshape(3, 1), vec_forward.reshape(3, 1)])
    T_world2bone[:3, :3] = rot_mat
    T_world2bone[:3, 3] = bone[0]
    
    return T_world2bone # (4, 4)


def bones2kinem(bones: torch.Tensor, parents: list):
    """
    - bones: (num_bones, 2, 3)
    - parents: list with length num_bones 
    """
    num_bones = len(bones)
    root_idx = parents.index(None)
    kinematics = torch.zeros(num_bones, 4, 4)
    
    for b in range(num_bones):
        if b != root_idx:
            T_world2bone = estimate_global_tsfm(bones[b].cpu())
            T_world2parent = estimate_global_tsfm(bones[parents[b]].cpu())
            T_parent2bone = torch.linalg.inv(T_world2parent) @ T_world2bone
            kinematics[b] = T_parent2bone

        else:
            T_world2root = estimate_global_tsfm(bones[b].cpu())
            kinematics[b] = T_world2root

    return kinematics.to(bones.device) # (num_bones, 4, 4)


def get_pose(kinematics: torch.Tensor):
    """
    - kinematics: (num_bones, 4, 4)
    """
    pose = matrix_to_axis_angle(kinematics[:, :3, :3])
    return pose # (num_bones, 3)


def lbs(verts: torch.Tensor, bind_kinem: torch.Tensor, poses: torch.Tensor, weights: torch.Tensor, parents: list, bone_scale: torch.Tensor=None):
    """
    Batched linear blend skinning, with bone scaling factors (if bone_scale is not None)

    - verts: (batch_size, num_verts, 3)
    - bind_kinem: (num_bones, 4, 4)
    - poses: (batch_size, num_bones, 3)
    - weights: (num_verts, num_bones)
    - bone_scale: (num_bones, 3) or None
    """
    
    num_bones = bind_kinem.shape[0]
    batch_size = poses.shape[0]
    root_idx = parents.index(None)

    # new kinematics with pose changed
    new_kinem = bind_kinem.repeat(batch_size, 1, 1, 1)
    new_kinem[..., :3, :3] = axis_angle_to_matrix(poses)

    # the translational part of bone b is scaled by the scaling factor of bone b's parent (the root bone unchanged)
    if bone_scale is not None:
        bone_scale = bone_scale.expand(batch_size, -1, -1)
        temp = copy.deepcopy(parents)
        temp[root_idx] = 0
        bone_scale_parent = bone_scale[:, temp] # (batch_size, num_bones, 3)
        bone_scale_parent[:, root_idx] = bone_scale[:, root_idx] 
        new_kinem[:, :, :3, 3] *= bone_scale_parent

    verts_h = torch.nn.functional.pad(verts, pad=[0, 1], value=1.).permute(0, 2, 1)
    verts_all = []
    # traverse the kinematic tree and compose all the transformations
    for j in range(num_bones):
        Bj = torch.eye(4).repeat(batch_size, 1, 1).to(verts.device)
        Tj = torch.eye(4).repeat(batch_size, 1, 1).to(verts.device)
        if bone_scale is not None:
            Sj = Scale(bone_scale[:, j]).get_matrix().to(verts.device)
            
        current_idx = j
        while True:
            Bj = bind_kinem[current_idx] @ Bj
            Tj = new_kinem[:, current_idx] @ Tj
            if current_idx == root_idx:
                break
            current_idx = parents[current_idx]

        if bone_scale is not None:
            Tj = Tj @ Sj
        Bj_inv = torch.linalg.inv(Bj)
        verts_j = Tj @ Bj_inv @ verts_h
        verts_all += [verts_j * weights[:, j]]

    new_verts = sum(verts_all).permute(0, 2, 1)[..., :3]

    return new_verts # (batch_size, num_verts, 3)


def get_new_bones(rest_bones: torch.Tensor, parents: list, poses: torch.Tensor, bone_scale: torch.Tensor=None):
    """
    Batched bone transformation with bone scaling

    - rest_bones: (num_bones, 2, 3)
    - parents: list with length num_bones
    - poses: (batch_size, num_bones, 3)
    - bone_scale: (num_boens, 3) 
    """
    num_bones = rest_bones.shape[0]
    batch_size = poses.shape[0]
    root_idx = parents.index(None)
    bind_kinem = bones2kinem(rest_bones, parents)

    # new kinematics with pose changed
    new_kinem = bind_kinem.repeat(batch_size, 1, 1, 1)
    new_kinem[..., :3, :3] = axis_angle_to_matrix(poses)

    # the translational part of bone b is scaled by the scaling factor of bone b's parent (the root bone unchanged)
    if bone_scale is not None:
        bone_scale = bone_scale.expand(batch_size, -1, -1)
        temp = copy.deepcopy(parents)
        temp[root_idx] = 0
        bone_scale_parent = bone_scale[:, temp]
        bone_scale_parent[:, root_idx] = bone_scale[:, root_idx]
        new_kinem[:, :, :3, 3] *= bone_scale_parent

    rest_bones_head = rest_bones[:, 0].repeat(batch_size, 1, 1)
    rest_bones_head = torch.nn.functional.pad(rest_bones_head, pad=[0, 1], value=1.).permute(0, 2, 1)
    rest_bones_tail = rest_bones[:, 1].repeat(batch_size, 1, 1)
    rest_bones_tail = torch.nn.functional.pad(rest_bones_tail, pad=[0, 1], value=1.).permute(0, 2, 1)
    new_bones = torch.zeros(batch_size, num_bones, 2, 3, device=rest_bones.device)
    
    # traverse the kinematic tree and compose all the transformations
    for j in range(num_bones):
        Bj = torch.eye(4).repeat(batch_size, 1, 1).to(rest_bones.device)
        Tj = torch.eye(4).repeat(batch_size, 1, 1).to(rest_bones.device)
        if bone_scale is not None:
            Sj = Scale(bone_scale[:, j]).get_matrix().to(rest_bones.device)
            
        current_idx = j
        while True:
            Bj = bind_kinem[current_idx] @ Bj
            Tj = new_kinem[:, current_idx] @ Tj
            if current_idx == root_idx:
                break
            current_idx = parents[current_idx]

        if bone_scale is not None:
            Tj = Tj @ Sj

        head_j = Tj @ torch.linalg.inv(Bj) @ rest_bones_head[..., j, None]
        tail_j = Tj @ torch.linalg.inv(Bj) @ rest_bones_tail[..., j, None]
        new_bones[:, j, 0] = head_j.permute(0, 2, 1)[..., :3].squeeze()
        new_bones[:, j, 1] = tail_j.permute(0, 2, 1)[..., :3].squeeze()

    return new_bones # (batch_size, num_bones, 2, 3)
