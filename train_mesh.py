import sys
sys.path.append('./mesh')
import torch
import math
import numpy as np
from tqdm import trange
from datetime import datetime
from torch.optim.lr_scheduler import _LRScheduler
from pytorch3d.transforms import matrix_to_axis_angle
from mesh.template import Template
from mesh.model import MeshDeformer
from mesh.render import MultiViewRenderer
from mesh.skinning import lbs, get_new_bones
from mesh.losses import *
from dataloader import MultiViewDataset, get_loader
from utils import procrustes, load_cameras, convert_cameras, pair_euclidean_nan


class CosinePerParamLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=None, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min if eta_min else [0.0] * len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Compute the learning rate for each parameter group.
        return [
            self.eta_min[i] + (base_lr - self.eta_min[i]) * 
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for i, base_lr in enumerate(self.base_lrs)
        ]


class MeshTrainer:
    def __init__(self, model: MeshDeformer, dataset: MultiViewDataset):
        self.model = model
        self.dataset = dataset
        self.gt = {}
        self.pred = {}
        self.config = None
        self.record = None

    # Learn scale and pscale on randomly sampled frames
    def run_stage1(self):
        n_sample = self.config['stage1']['n_sample']
        n_iter = self.config['stage1']['n_iter']
        scale_lr_init = self.config['stage1']['scale_lr_init']
        scale_lr_min = self.config['stage1']['scale_lr_min']
        rigid_lr_init = self.config['stage1']['rigid_lr_init']
        rigid_lr_min = self.config['stage1']['rigid_lr_min']
        pose_lr_init = self.config['stage1']['pose_lr_init']
        pose_lr_min = self.config['stage1']['pose_lr_min']
        pscale_lr_init = self.config['stage1']['pscale_lr_init']
        pscale_lr_min = self.config['stage1']['pscale_lr_min']
        tex_lr_init = self.config['stage1']['tex_lr_init']
        tex_lr_min = self.config['stage1']['tex_lr_min']
        c_kpt3d = self.config['stage1']['c_kpt3d']
        c_kpt2d = self.config['stage1']['c_kpt2d']
        c_mask = self.config['stage1']['c_mask']
        c_iou = self.config['stage1']['c_iou']
        c_rgb = self.config['stage1']['c_rgb']
        c_pose_reg = self.config['stage1']['c_pose_reg']
        c_pose_lim = self.config['stage1']['c_pose_lim']
        c_pscale_smt = self.config['stage1']['c_smt']
        c_pscale_reg = self.config['stage1']['c_pscale_reg']
        c_pscale_iso = self.config['stage1']['c_pscale_iso']
        c_rigid_smooth = self.config['stage1']['c_rigid_smooth']
        c_arap = self.config['stage1']['c_arap']

        
        self.model.set_active(['scale', 'rigid', 'pose', 'pscale', 'tex'])
        parameters = [{'params': self.model.param_dict['scale'], 'lr': scale_lr_init},
                      {'params': self.model.param_dict['rigid'], 'lr': rigid_lr_init},
                      {'params': self.model.param_dict['pose'], 'lr': pose_lr_init},
                      {'params': self.model.param_dict['pscale'], 'lr': pscale_lr_init},
                      {'params': self.model.param_dict['tex'], 'lr': tex_lr_init}]

        dataloader = get_loader(dataset=self.dataset, batch_size=self.model.batch_size, shuffle=True) # random sample 
        dataiter = iter(dataloader)

        print('Learning global parameters:')
        for i in range(n_sample):
            self.model.reset(to_reset=['pose'])
            self.get_gt(next(dataiter))
            self.init_rigid(self.gt['kpt3d'])
            optimizer = torch.optim.Adam(params=parameters)
            scheduler = CosinePerParamLR(optimizer=optimizer, T_max=n_iter, eta_min=[scale_lr_min, rigid_lr_min, pose_lr_min, pscale_lr_min, tex_lr_min])

            loop = trange(n_iter)
            loop.set_description(f'[Sample {i+1}/{n_sample}]')
            for _ in loop:
                optimizer.zero_grad()
                self.get_pred(self.model())
                loss = (c_kpt3d * loss_kpt3d(self.pred['kpt3d'], self.gt['kpt3d'], self.kpt_weight) +
                        c_kpt2d * loss_kpt2d(self.pred['kpt2d'], self.gt['kpt2d'], self.kpt_weight) +
                        c_iou * loss_iou(self.pred['mask'], self.gt['mask']) +
                        c_mask * loss_mask(self.pred['mask'], self.gt['mask']) + 
                        c_rgb * loss_rgb(self.pred['rgb'], self.gt['rgb'], self.pred['mask'], self.gt['mask']) +
                        c_pose_reg * loss_pose_reg(self.model.pose) +
                        c_pose_lim * loss_pose_lim(self.model.pose, self.model.template.bind_pose, self.lim) +
                        c_pscale_smt * loss_pscale_smt(self.model.pscale, self.smt) +
                        c_pscale_reg * loss_pscale_reg(self.model.pscale) +
                        c_pscale_iso * loss_pscale_iso(self.model.pscale) + 
                        c_rigid_smooth * loss_rigid_smooth(self.model.rigid) +
                        c_arap * loss_arap(self.model.template, self.pred['mesh'])
                        )
                loss.backward()
                optimizer.step()
                scheduler.step()

        self.eval()

    
    # Construct a new template with learnt scale, pscale and texture parameters
    def adjust_template(self):
        with torch.no_grad():
            new_verts = lbs(verts=self.model.template.mesh.verts_padded(),
                            bind_kinem=self.model.template.bind_kinem,
                            poses=self.model.template.bind_pose[None, ...],
                            weights=self.model.template.skin_weights,
                            parents=self.model.template.parents,
                            bone_scale=torch.exp(self.model.pscale))
            new_verts *= torch.exp(self.model.scale)

            new_bones = get_new_bones(rest_bones=self.model.template.bones,
                                      parents=self.model.template.parents,
                                      poses=self.model.template.bind_pose[None, ...],
                                      bone_scale=torch.exp(self.model.pscale))
            new_bones = new_bones[0] * torch.exp(self.model.scale)
            
            new_tex = nn.Sigmoid()(self.model.tex)[None, ...]
            
            self.model.template.update(new_verts, new_bones, new_tex)


    # Fit all frames, with scale, texture and pscale fixed
    def run_stage2(self):
        # read config 
        n_iter = self.config['stage2']['n_iter']
        rigid_lr_init = self.config['stage2']['rigid_lr_init']
        rigid_lr_min = self.config['stage2']['rigid_lr_min']
        pose_lr_init = self.config['stage2']['pose_lr_init']
        pose_lr_min = self.config['stage2']['pose_lr_min']
        c_kpt3d = self.config['stage2']['c_kpt3d']
        c_kpt2d = self.config['stage2']['c_kpt2d']
        c_mask = self.config['stage2']['c_mask']
        c_iou = self.config['stage2']['c_iou']
        c_pose_reg = self.config['stage2']['c_pose_reg']
        c_pose_lim = self.config['stage2']['c_pose_lim']
        c_pose_smooth = self.config['stage2']['c_pose_smooth']
        c_rigid_smooth = self.config['stage2']['c_rigid_smooth']
        c_arap = self.config['stage2']['c_arap']


        self.model.set_active(['rigid', 'pose'])
        parameters = [{'params': self.model.param_dict['rigid'], 'lr': rigid_lr_init},
                      {'params': self.model.param_dict['pose'], 'lr': pose_lr_init}]

        print('Mesh fitting:')
        dataloader = get_loader(dataset=self.dataset, batch_size=self.model.batch_size, shuffle=False) # in order 
        batch_count = 0
        for out in dataloader:
            self.get_gt(out)
            self.init_rigid(self.gt['kpt3d'], scale=False)
            optimizer = torch.optim.Adam(params=parameters)
            scheduler = CosinePerParamLR(optimizer=optimizer, T_max=n_iter, eta_min=[rigid_lr_min, pose_lr_min])
            batch_count += 1

            loop = trange(n_iter)
            loop.set_description(f'[Mesh fitting {batch_count}/{len(dataloader)}]')
            for _ in loop:
                optimizer.zero_grad()
                self.get_pred(self.model())
                loss = (c_kpt3d * loss_kpt3d(self.pred['kpt3d'], self.gt['kpt3d'], self.kpt_weight) +
                        c_kpt2d * loss_kpt2d(self.pred['kpt2d'], self.gt['kpt2d'], self.kpt_weight) +
                        c_mask * loss_mask(self.pred['mask'], self.gt['mask']) + # old c_mask
                        c_iou * loss_iou(self.pred['mask'], self.gt['mask']) + 
                        c_pose_reg * loss_pose_reg(self.model.pose) +
                        c_pose_lim * loss_pose_lim(self.model.pose, self.model.template.bind_pose, self.lim) + 
                        c_pose_smooth * loss_pose_smooth(self.model.pose) +
                        c_rigid_smooth * loss_rigid_smooth(self.model.rigid) +
                        c_arap * loss_arap(self.model.template, self.pred['mesh'])
                        )
                loss.backward()
                optimizer.step()
                scheduler.step()

            self.eval()
            self.record_batch()
            self.model.reset(to_reset=['pose', 'rigid']) 


    def eval(self):
        with torch.no_grad():
            IoU = 1 - loss_iou(self.gt['mask'], self.pred['mask'])
            MPJPE = pair_euclidean_nan(self.gt['kpt3d'][:, self.kpt_eval], self.pred['kpt3d'][:, self.kpt_eval])
            MPJPE = MPJPE * self.world_scale_factor
            print(f'IoU: {IoU.item():.3f}, MPJPE: {MPJPE.item():.3f}\n')


    def record_batch(self):
        with torch.no_grad():
            if self.record is None:
                self.record = {}
                self.record['v'] = self.model.template.mesh.verts_packed().detach().cpu()
                self.record['f'] = self.model.template.mesh.faces_packed().detach().cpu()
                self.record['c'] = self.model.template.mesh.textures.verts_features_packed().detach().cpu()
                self.record['bones'] = self.model.template.bones.detach().cpu()
                self.record['skin_weights'] = self.model.template.skin_weights.detach().cpu()
                self.record['mapping_matrix'] = self.model.template.mapping_matrix.detach().cpu()
                self.record['parents'] = self.model.template.parents
                self.record['pose'] = []
                self.record['rigid'] = []

            self.record['pose'].append(self.model.pose.detach().cpu())
            self.record['rigid'].append(self.model.rigid.detach().cpu())


    def save(self):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        folder_path = os.path.join(self.config['save_path'], timestamp)
        os.makedirs(folder_path, exist_ok=True)
        self.record['pose'] = torch.cat(self.record['pose'], dim=0)
        self.record['rigid'] = torch.cat(self.record['rigid'], dim=0)
        torch.save(self.record, os.path.join(folder_path, 'fitting.pth'))


    def train(self, config):
        self.config = config
        self.world_scale_factor = 1 / config['world_scale']
        self.kpt_weight = torch.tensor(config['kpt_weight']).to(self.model.device)
        self.kpt_eval = config['kpt_eval']
        self.smt = config['smt']
        self.lim = torch.tensor(config['lim']).to(self.model.device)
        self.run_stage1()
        self.adjust_template()
        self.model.reset()
        self.run_stage2()
        self.save()

    def get_gt(self, out):
        self.gt['rgb'] = out[0].to(self.model.device)
        self.gt['mask'] = out[1].to(self.model.device)
        self.gt['kpt2d'] = out[2].to(self.model.device)
        self.gt['kpt3d'] = out[3].to(self.model.device)

    def get_pred(self, out):
        self.pred['mesh'] = out[0].to(self.model.device)
        self.pred['rgb'] = out[1].to(self.model.device)
        self.pred['mask'] = out[2].to(self.model.device)
        self.pred['kpt2d'] = out[3].to(self.model.device)
        self.pred['kpt3d'] = out[4].to(self.model.device)

    def init_rigid(self, kpts3d, scale=True):
        # kpt3d: (b, nk, 3)
        with torch.no_grad():
            kpts_template = self.model.template.mapping_matrix @ self.model.template.mesh.verts_packed() # (nk, 3)
            mean_s = 0.
            for i in range(self.model.batch_size):
                kpts = kpts3d[i]
                mask = torch.isnan(kpts).sum(1) == 0
                s, R, t = procrustes(kpts_template[mask].T, kpts[mask].T)
                mean_s += s
                aa = matrix_to_axis_angle(R)
                self.model.rigid[i, :3], self.model.rigid[i, 3:] = t.squeeze(), aa.reshape(-1, 3)

            if scale:
                self.model.scale[:] = torch.log(mean_s / self.model.batch_size)



import argparse
import yaml
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_path = config['dataset']
    cam_path = os.path.join(dataset_path, 'cameras.npy')
    template_path = os.path.join(dataset_path, 'template.pth')
    wh = eval(config['image_size']) # wh
    hw = (wh[1], wh[0])
    device = config['device']
    world_scale = config['world_scale']
    batch_size = config['batch_size']

    template = Template(template_path, device=device)
    cameras = load_cameras(cam_path)
    whs = np.array(wh)[None, ...].repeat(cameras.nc, axis=0)
    cameras.scale(new_sizes=whs, world_scale=world_scale)
    cameras = convert_cameras(cameras)
    renderer = MultiViewRenderer(image_size=hw, cameras=cameras, lights=None, device=device, faces_per_pixel=5, sigma=1e-6)
    model = MeshDeformer(batch_size=batch_size, template=template, renderer=renderer, device=device)
    dataset = MultiViewDataset(root_path=dataset_path, resize=hw, mask_invert=False, world_scale=world_scale)

    trainer = MeshTrainer(model=model, dataset=dataset)
    trainer.train(config)

    