import pytorch3d
import torch
import math
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_euler_angles


def loss_kpt3d(x, y, kpt_weight):
   return torch.nanmean((x - y).abs() * kpt_weight.reshape(1, -1, 1))

def loss_kpt2d(x, y, kpt_weight):
   return torch.nanmean((x - y).abs() * kpt_weight.reshape(1, 1, -1, 1))


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
dbce = DiceBCELoss()
def loss_dbce(x, y):
   return dbce(x, y)

def loss_mask(x, y):
   return (x - y).square().mean() 

def loss_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    mask1 = mask1.bool()
    mask2 = mask2.bool()
    
    intersection = (mask1 & mask2).sum(dim=(-2, -1))  # Sum over H, W
    union = (mask1 | mask2).sum(dim=(-2, -1))  # Sum over H, W
    iou = intersection.float() / (union.float() + 1e-6)  # Avoid division by zero
    loss = 1 - iou.mean()
    
    return loss
    


def loss_rgb(x, y, x_mask, y_mask):
   mask = x_mask.detach() * y_mask.detach()
   diff = (x - y).abs() * mask
   return diff.sum() / mask.sum()


def loss_pscale_smt(x, smt):
   loss = 0.
   if len(smt) == 0:
    return loss
   
   else:
    for i in range(len(smt)):
        group = smt[i]
        loss += torch.diff(torch.exp(x)[group], dim=0).square().mean()

    return loss / len(smt)
  
   

def loss_pose_reg(x):
   return torch.mean(x ** 2)

def loss_pscale_reg(x):
   return torch.mean((torch.exp(x) - 1) ** 2)

def loss_pscale_iso(x):
   return (torch.diff(torch.exp(x)) ** 2).mean()

def loss_pose_smooth(x):
   if len(x) == 1:
      return 0.
   else:
      return torch.mean(torch.diff(x, dim=0) ** 2)

def loss_rigid_smooth(x):
   if len(x) == 1:
      return 0.
   else:
      trans = x[:, :3]
      l_trans = torch.mean(torch.diff(trans, dim=0) ** 2)
      rot = axis_angle_to_quaternion(x[:, 3:])
      dot_prod = (rot[:, 1:] * rot[:, :-1]).sum(dim=-1)
      l_rot = torch.mean(1 - dot_prod ** 2)
      return l_trans + l_rot

# Arap loss
def edges_to_sparse_incidence(edges, num_vertices):
    num_edges = edges.shape[0]
    row_indexes = torch.arange(num_edges, dtype=torch.long, device=edges.device).repeat_interleave(2)
    col_indexes = edges.reshape(-1)
    indexes = torch.stack([row_indexes, col_indexes])
    values = torch.FloatTensor([1, -1]).to(edges.device).repeat(num_edges)
    return torch.sparse.FloatTensor(indexes, values, torch.Size([num_edges, num_vertices]))


def compute_svd_rotation(vertices_rest_pose, vertices_deformed_pose, incidence_mat):
    """
    Adapted from:
    https://github.com/kzhou23/shape_pose_disent/blob/a8017c405892c98f52fa9775327172633290b1d8/arap.py#L76

    vertices_rest_pose: B x V x D
    vertices_deformed_pose: B x V x D
    incidence_mat: E x V
    
    """
    batch_size, num_vertices, dimensions = vertices_rest_pose.shape
    vertices = torch.cat((vertices_rest_pose, vertices_deformed_pose), dim=0)
    # 2B x V x D -> V x (D x 2B)
    vertices = rearrange(vertices, 'a v d -> v (d a)')
    # E x V . V x (D x 2B) - > E x (D x 2B)
    edges = torch.sparse.mm(incidence_mat, vertices)
    edges = rearrange(edges, 'e (d a) -> a e d', d=dimensions)                 
    rest_edges, deformed_edges = torch.split(edges, batch_size, dim=0)

    edges_outer = torch.matmul(rest_edges[:, :, :, None], deformed_edges[:, :, None, :])
    edges_outer = rearrange(edges_outer, 'b e d1 d2 -> e (b d1 d2)')

    abs_incidence_mat = incidence_mat.clone()
    abs_incidence_mat._values()[:] = torch.abs(abs_incidence_mat._values())
    
    # transposed S
    S = torch.sparse.mm(abs_incidence_mat.t(), edges_outer)
    S = rearrange(S, 'v (b d1 d2) -> b v d2 d1', v=num_vertices, b=batch_size, d1=dimensions, d2=dimensions)
    
    # SVD on gpu is extremely slow! https://github.com/pytorch/pytorch/pull/48436
    device = S.device
    U, _, V = torch.svd(S.cpu())
    U = U.to(device)
    V = V.to(device)

    det_sign = torch.det(torch.matmul(U, V.transpose(-2, -1)))
    U = torch.cat([U[..., :-1], U[..., -1:] * det_sign[..., None, None]], axis=-1)

    rotations = torch.matmul(U, V.transpose(-2, -1))

    return rotations


def compute_rotation(vertices_rest_pose, vertices_deformed_pose, edges):
  num_vertices = vertices_rest_pose.shape[1]
  incidence_mat = edges_to_sparse_incidence(edges, num_vertices)
  rot = compute_svd_rotation(vertices_rest_pose, vertices_deformed_pose, incidence_mat)
  rot = pytorch3d.transforms.matrix_to_quaternion(rot)
  return rot


def quaternion_normalize(quaternion, eps=1e-12):
  return l2_normalize(quaternion, dim=-1, epsilon=eps)


def l2_normalize(x, dim=-1, epsilon=1e-12):
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.clamp(square_sum, min=epsilon))
    return x * x_inv_norm
    

def arap_energy(vertices_rest_pose,
           vertices_deformed_pose,
           quaternions,
           edges,
           vertex_weight=None,
           edge_weight=None,
           conformal_energy=True,
           aggregate_loss=True):

    if not conformal_energy:
      quaternions = quaternion_normalize(quaternions)
    # Extracts the indices of vertices.
    indices_i, indices_j = torch.unbind(edges, dim=-1)
    # Extracts the vertices we need per term.
    vertices_i_rest = vertices_rest_pose[..., indices_i, :]
    vertices_j_rest = vertices_rest_pose[..., indices_j, :]
    vertices_i_deformed = vertices_deformed_pose[..., indices_i, :]
    vertices_j_deformed = vertices_deformed_pose[..., indices_j, :]
    # Extracts the weights we need per term.
    weights_shape = vertices_i_rest.shape[-2]
    if vertex_weight is not None:
      weight_i = vertex_weight[indices_i]
      weight_j = vertex_weight[indices_j]
    else:
      weight_i = weight_j = torch.ones(weights_shape, dtype=vertices_rest_pose.dtype, device=vertices_rest_pose.device)
    weight_i = weight_i[..., None]
    weight_j = weight_j[..., None]
    if edge_weight is not None:
      weight_ij = edge_weight
    else:
      weight_ij = torch.ones(weights_shape, dtype=vertices_rest_pose.dtype, device=vertices_rest_pose.device)
    weight_ij = weight_ij[..., None]
    # Extracts the rotation we need per term.
    quaternion_i = quaternions[..., indices_i, :]
    quaternion_j = quaternions[..., indices_j, :]
    # Computes the energy.
    deformed_ij = vertices_i_deformed - vertices_j_deformed
    rotated_rest_ij = pytorch3d.transforms.quaternion_apply(quaternion_i, (vertices_i_rest - vertices_j_rest))
    energy_ij = weight_i * weight_ij * (deformed_ij - rotated_rest_ij)
    deformed_ji = vertices_j_deformed - vertices_i_deformed
    rotated_rest_ji = pytorch3d.transforms.quaternion_apply(quaternion_j, (vertices_j_rest - vertices_i_rest))
    energy_ji = weight_j * weight_ij * (deformed_ji - rotated_rest_ji)
    energy_ij_squared = torch.sum(energy_ij ** 2, dim=-1)
    energy_ji_squared = torch.sum(energy_ji ** 2, dim=-1)
    if aggregate_loss:
      average_energy_ij = torch.mean(energy_ij_squared, dim=-1)
      average_energy_ji = torch.mean(energy_ji_squared, dim=-1)
      return (average_energy_ij + average_energy_ji) / 2.0
    return torch.cat((energy_ij_squared, energy_ji_squared), dim=-1)


def loss_arap(template, new_meshes):
    # squash batch dimensions
    vertices_rest_pose = template.extended_mesh.verts_padded()
    vertices_deformed_pose = new_meshes.verts_padded()
    edges = template.mesh.edges_packed()
    vertices_rest_pose_shape = list(vertices_rest_pose.shape)
    vertices_deformed_pose_shape = list(vertices_deformed_pose.shape)
    vertices_rest_pose = vertices_rest_pose.reshape([-1] + vertices_rest_pose_shape[-2:])
    vertices_deformed_pose = vertices_deformed_pose.reshape([-1] + vertices_deformed_pose_shape[-2:])
    
    # try:
    quaternions = compute_rotation(vertices_rest_pose, vertices_deformed_pose, edges)
    # except RuntimeError:
    #   print('SVD did not converge')
    # batch_size = vertices_rest_pose.shape[0]
    # num_vertices = vertices_rest_pose.shape[-2]
    # quaternions = pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.euler_angles_to_matrix(torch.zeros([batch_size, num_vertices, 3], device=vertices_rest_pose.device), 'XYZ'))
    
    quaternions = quaternions.detach()

    energy = arap_energy(
      vertices_rest_pose,
      vertices_deformed_pose,
      quaternions,
      edges,
      aggregate_loss=True,
      conformal_energy=False)
    return energy.reshape(vertices_rest_pose_shape[:-2]).mean()


def loss_pose_lim(new_pose, bind_pose, limit):
    """
    new_pose: (B, N, 3) axis-angle
    bind_pose: (N, 3)
    limit: (N, 3, 2) in degrees
    """
    B, N, _ = new_pose.shape
    R_bind = axis_angle_to_matrix(bind_pose)  # (N, 3, 3)
    R_new = axis_angle_to_matrix((bind_pose + new_pose).view(-1, 3))  # (B*N, 3, 3)

    bind_angles = rad2deg(matrix_to_euler_angles(R_bind, convention='XYZ'))  # (N, 3)
    new_angles = rad2deg(matrix_to_euler_angles(R_new, convention='XYZ')).view(B, N, 3)  # (B, N, 3)
    angle_delta = new_angles - bind_angles.unsqueeze(0)  # (B, N, 3)
    angle_min = limit[..., 0].unsqueeze(0)  # (1, N, 3)
    angle_max = limit[..., 1].unsqueeze(0)  # (1, N, 3)

    penalty = torch.where(
        angle_delta < angle_min, angle_min - angle_delta,
        torch.where(angle_delta > angle_max, angle_delta - angle_max, torch.zeros_like(angle_delta))
    )

    loss = penalty.abs().mean()
    return loss


def rad2deg(x):
    return x * (180.0 / math.pi)

