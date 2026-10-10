import pytorch3d
import torch
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from skinning import *
from vis import *


class Template:
    def __init__(self, config_path, device) -> None:
        self.device = device
        config = torch.load(config_path)

        # construct pytorch3d mesh
        v = config['v'].to(device) # (num_verts, 3)
        f = config['f'].to(device) # (num_faces, 3)
        self.mesh: Meshes = pytorch3d.structures.Meshes(verts=[v], faces=[f])
        self.mesh.textures = TexturesVertex(torch.full_like(v, 0.5)[None, ...]) # paint gray
        self.mesh_o3d = p3d2o3d(self.mesh, compute_norm=True) # construct open3d mesh
        self.extended_mesh = None

        self.bones: torch.FloatTensor = config['bones'].to(device) # rest bones coordinates
        self.parents: list = config['parents'] # parents specifying bone connections
        self.skin_weights: torch.FloatTensor = config['skin_weights'].to(device) # skinning weights used in blend skinning
        self.mapping_matrix: torch.FloatTensor = config['mapping_matrix'].to(device) # mapping matrix that maps from mesh vertices to keypoints
        self.bind_kinem = bones2kinem(self.bones, self.parents)
        self.bind_pose = get_pose(self.bind_kinem)

        self.root_idx = self.parents.index(None)
        self.num_bones = self.bones.shape[0]
        self.num_keypoints = self.mapping_matrix.shape[0]
        self.num_verts = self.mapping_matrix.shape[1]
        self.num_edges = self.mesh.edges_packed().shape[0]
    


    def show_mesh(self):
        o3d.visualization.draw_geometries([self.mesh_o3d], window_name='Template')

    def print_kinem(self):
        print_kinematics(self.bind_kinem.cpu(), self.parents)

    def show_bones(self, bone_thickness=0.5, special_idx=[]):
        print_kinematics(self.bind_kinem, self.parents)
        bone_meshes, frame_meshes = plot_bones(self.bones.cpu(), 
                                               self.parents, 
                                               visualize=False, 
                                               thickness=bone_thickness, 
                                               special_idx=special_idx)
        
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh_o3d)
        o3d.visualization.draw_geometries([wireframe, bone_meshes, frame_meshes], window_name='Rest bones')

    def show_skin_weights(self):
        show_weights(self.mesh, self.skin_weights)

    def show_keypoints(self, dot_size=0.1):
        mapped_kpts = self.mapping_matrix @ self.mesh.verts_packed()
        plot_kpt3d(mapped_kpts, self.mesh, dot_size=dot_size)

    def show_mapping_weights(self, temperature=0.001):
        show_mat = torch.nn.functional.softmax(self.mapping_matrix.T / temperature, dim=1)
        show_weights(self.mesh, show_mat, compute_norm=True)

    def extend_mesh(self, batch_size):
        self.extended_mesh = self.mesh.extend(batch_size)

    # update with new verts and new bones
    def update(self, new_verts, new_bones, new_tex=None):
        self.mesh = self.mesh.update_padded(new_verts)
        if new_tex is not None:
            self.mesh.textures = TexturesVertex(new_tex)
        self.mesh_o3d = p3d2o3d(self.mesh, compute_norm=True)
        if self.extended_mesh is not None:
            self.extended_mesh = self.mesh.extend(len(self.extended_mesh))

        self.bones = new_bones
        self.bind_kinem = bones2kinem(self.bones, self.parents)
        self.bind_pose = get_pose(self.bind_kinem)
