import open3d as o3d
import numpy as np
import pytorch3d
import pytorch3d.structures
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from pytorch3d.renderer import TexturesVertex
from torchvision.utils import make_grid, draw_keypoints
from torchvision.transforms.functional import convert_image_dtype
from skinning import estimate_global_tsfm



def show_images(imgs: torch.Tensor, nrow=8, title=None):
    if imgs.dim() == 3: # (c, h, w)
        imglist = [imgs]
    else: # (b, c, h, w)
        imgs = make_grid(imgs, nrow=nrow)
        imglist = [imgs]
    plt.rcParams["savefig.bbox"] = 'tight'
    _, axs = plt.subplots(ncols=len(imglist), squeeze=False)
    for i, img in enumerate(imglist):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if title is not None:
        plt.title(title)

    plt.show()
    

def o3d2p3d(mesh: o3d.geometry.TriangleMesh, device='cpu') -> pytorch3d.structures.Meshes:
    # only support meshes with verts_colors
    verts = torch.FloatTensor(np.asarray(mesh.vertices)).to(device)
    faces = torch.FloatTensor(np.asarray(mesh.triangles)).to(device)
    mesh_p3d = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
    if len(mesh.vertex_colors):
        colors = torch.FloatTensor(np.asarray(mesh.vertex_colors)).to(device)
    else:
        colors = torch.full_like(verts, 0.5, device=device)
    mesh_p3d.textures = TexturesVertex(verts_features=colors[None, ...])
    return mesh_p3d


def p3d2o3d(mesh: pytorch3d.structures.Meshes, compute_norm=False) -> o3d.geometry.TriangleMesh:
    # only support meshes with TexturesVertex, otherwise color will be set to gray
    verts = o3d.utility.Vector3dVector(mesh.verts_packed().cpu().numpy())
    faces = o3d.utility.Vector3iVector(mesh.faces_packed().cpu().numpy())
    mesh_o3d = o3d.geometry.TriangleMesh(vertices=verts, triangles=faces)
    if type(mesh.textures) is TexturesVertex:
        colors = o3d.utility.Vector3dVector(mesh.textures.verts_features_packed().cpu().numpy())
        mesh_o3d.vertex_colors = colors
    else:
        colors = o3d.utility.Vector3dVector((torch.full_like(mesh.verts_packed(), 0.5)).cpu().numpy())
        mesh_o3d.vertex_colors = colors

    if compute_norm:
        mesh_o3d.compute_vertex_normals()

    return mesh_o3d


def show_mesh(mesh: pytorch3d.structures.Meshes, compute_norm=True):
    mesh = p3d2o3d(mesh)
    if compute_norm:
        mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], 'Mesh')


def show_weights(mesh: pytorch3d.structures.Meshes, weights, compute_norm=False, color_map=None):
    if color_map is None:
        color_map = torch.rand(weights.shape[-1], 3).to(weights.device)
    color = (weights @ color_map).cpu().numpy()
    weight_mesh = p3d2o3d(mesh)
    weight_mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    if compute_norm:
        weight_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([weight_mesh], 'Weights')


def print_kinematics(kinematics: torch.Tensor, parents: list):
    num_bones = len(parents)
    root_idx = parents.index(None)
    print(f'kinematic chain with {num_bones} bones, [bone {root_idx}] is the root\n')
    for i in range(num_bones):
        print(f'bone {i}:')
        print(f'parent: [bone {parents[i]}]')
        print(f'transform:\n{kinematics[i].cpu().numpy()}')
        print('-' * 100)
        print('-' * 100)


def plot_bones(bones, parents, visualize=True, show_world_frame=False, special_idx=[], thickness=1.0, mesh=None):
    """
    - bones: (num_bones, 2, 3) tensor, with bones[b, 0] being the head of bone b and bones[b, 1] being the tail
    - parents: (num_bones, ) list
    - mesh: pytorch3d Meshes, show bone along with wireframe if it is not None
    """
    num_bones = len(parents)
    bone_lengths = np.linalg.norm(bones[:, 0] - bones[:, 1], axis=1)
    average_length = np.mean(bone_lengths)


    all_meshes = []
    bone_meshes = None
    frame_meshes = None
    if show_world_frame:
        all_meshes += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=average_length / 2, origin=np.array([0.0, 0.0, 0.0]))]
    
    for b in range(num_bones):
        # create the bone's mesh at the origin of the world frame
        bone_mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=thickness*(average_length/10), 
                                                           cone_radius=thickness*(average_length/10),
                                                           cylinder_height=bone_lengths[b] * 0.9,
                                                           cone_height=bone_lengths[b] * 0.1)
        bone_mesh.compute_vertex_normals()
        if b in special_idx:
            # paint the bone mesh with red if it's in the special list
            bone_mesh.paint_uniform_color([1, 0, 0.])
        else:
            # else paint it with blue
            bone_mesh.paint_uniform_color([0, 0, 1.])
        # move the bone to its place using transformation matrix
        T_world2bone = estimate_global_tsfm(bones[b]).cpu().numpy()
        bone_mesh.transform(T_world2bone)
        if bone_meshes is not None:
            bone_meshes += bone_mesh
        else:
            bone_meshes = bone_mesh
        
        # create the mesh of the bone's local frame at the origin of the world frame and then transform it
        frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=average_length / 3, 
                                                                       origin=np.array([0.0, 0.0, 0.0]))
        frame_mesh.compute_vertex_normals()
        frame_mesh.transform(T_world2bone)
        if frame_meshes is not None:
            frame_meshes += frame_mesh
        else:
            frame_meshes = frame_mesh

        if mesh is not None:
            mesh_o3d = p3d2o3d(mesh)
            wire_frame = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_o3d)

    # show the meshes in an o3d window if to visualize
    if visualize:
        all_meshes = all_meshes + [bone_meshes] + [frame_meshes] + [wire_frame]
        o3d.visualization.draw_geometries(all_meshes)

    return bone_meshes, frame_meshes


def plot_kpt2d(keypoints: torch.Tensor, images: torch.Tensor, color='red', dot_size=1.):
    """
    - keypoints: (batch_size, num_kpts, 2)
    - images: (batch_size, c, h, w)
    - color: 'red', 'blue', ...
    - dot_size: ...
    """
    kpt_imgs = []
    for i in range(len(images)):
        image = images[i]
        kpt = keypoints[i]
        mask = torch.isnan(kpt).sum(-1) == 0
        kpt = kpt[mask][None, ...]
        kpt_img = draw_keypoints(convert_image_dtype(image, torch.uint8), kpt, colors=color, radius=dot_size)
        kpt_imgs.append(kpt_img)
    kpt_imgs = torch.stack(kpt_imgs)
    show_images(kpt_imgs)


def plot_kpt3d(keypoints: torch.Tensor, mesh: pytorch3d.structures.Meshes=None, colors=None, dot_size=1):
    """
    - mesh: pytorch3d meshes object
    - keypoints: (num_kpts, 3) tensor
    - colors: (num_kpts, 3) tensor specifying the color of each keypoint, or (3, ) with all one color for all the keypoints
    - dot_size: radius of the plotted keypoint
    - visualize: whether to show plot in an o3d window
    """
    if mesh is not None:
        mesh_o3d = p3d2o3d(mesh)
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_o3d)
    num_kpts = keypoints.shape[0]
    dot_meshes = None
    if colors is None:
        colors = torch.tensor([1., 0., 0.]).repeat(num_kpts, 1).tolist()
    elif colors.shape[0] != num_kpts:
        colors = colors.repeat(num_kpts, 1).tolist()

    for i in range(num_kpts):
        kpt = keypoints[i]
        dot_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=dot_size)
        dot_mesh.paint_uniform_color(colors[i])
        #dot_mesh.compute_vertex_normals()
        dot_mesh.translate(kpt.cpu().numpy().reshape(3, 1))
        if dot_meshes is None:
            dot_meshes = dot_mesh
        else:
            dot_meshes += dot_mesh

    all_meshes = [dot_meshes]
    if mesh is not None:
        dot_meshes = all_meshes.append(wireframe)
    
    o3d.visualization.draw_geometries(all_meshes, window_name='Keypoints')