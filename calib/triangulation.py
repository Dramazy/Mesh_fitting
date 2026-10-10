import warnings
import copy
import random
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from itertools import combinations
from cameras import Cameras, pair_euclidean_nan


def triangulate(cameras: Cameras, p2d: np.ndarray) -> np.ndarray:
    """
    Triangulation by minimizing reprojection error using SVD

    - projection_mats: (nc, 3, 4) 
    - points2d: (nc, nk, 2)
    """
    nk = p2d.shape[1]
    p2d = cameras.undistort_points(p2d) # undistorted pixel coordinates
    points3d = np.full((nk, 3), np.nan)
    for i in range(nk):
        points = p2d[:, i]
        visible = np.where(np.isnan(points).sum(axis=1) == 0)
        nv = len(visible[0]) # num of visible views
        # triangulate if point is seen at least in two views
        if nv >= 2:
            sub_cam = cameras.P[visible]
            sub_kpt = points[visible]

            # solve the linear system using svd
            A = np.zeros((2*nv, 4))
            for j in range(nv):
                x, y = sub_kpt[j, 0], sub_kpt[j, 1]
                A[2*j] = x * sub_cam[j, 2] - sub_cam[j, 0]
                A[2*j+1] = y * sub_cam[j, 2] - sub_cam[j, 1]

            _, _, v = np.linalg.svd(A)
            p3d = v[-1]
            points3d[i] = p3d[:3] / p3d[3]

    return points3d


def triangulate_robust(cameras: Cameras, p2d: np.ndarray, thresh: float = 10.0, max_samples: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D points by:
    1. Triangulating all (or max_samples) 2-view combinations to get candidates.
    2. For each candidate, mark a view as an inlier if its rpje on that view ≤ threshold.
    3. For each point, the best candidate is the one that has the most inliers
    4. If multiple candidates have the same number of inliers, choose the one with the smallest average rpje on inliers
    5. Refine 3d locations using gradient based methods

    - cameras: Cameras object
    - p2d: (nc, nk, 2)
    - thresh: threshold for reprojection error, above it are considered as outliers
    - max_samples: max number for 2-view combinations
    - verbose: ...
    """
    nk = p2d.shape[1]
    combs = combinations([i for i in range(cameras.nc)], 2)
    if max_samples is not None and max_samples < cameras.nc:
        combs = random.sample(list(combs), max_samples)
    candidates = []
    rpjes = []
    outliers = []
    # triangulate on all (or max_samples) possible 2-views 
    for comb in combs:
        comb = list(comb)
        candidate = triangulate(cameras[comb], p2d[comb]) # (nk, 3)
        rpj = cameras.project_points(candidate) # (nc, nk, 2)
        rpje = pair_euclidean_nan(rpj, p2d, mean=False) # (nc, nk)

        # reprojection errors > threshold are considered as outliers
        is_outlier = rpje > thresh
        rpje[is_outlier] = np.nan

        candidates.append(candidate)
        rpjes.append(rpje)
        outliers.append(is_outlier)

    candidates = np.array(candidates) # (ncomb, nk, 3)
    rpjes = np.array(rpjes) # (ncomb, nc, nk)
    outliers = np.array(outliers) # (ncomb, nc, nk)

    n_inliers = (~np.isnan(rpjes)).sum(axis=1) # (ncomb, nk)
    not_most_inliers = np.where(n_inliers != n_inliers.max(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        comb_rpje = np.nanmean(rpjes, axis=1) # (ncmob, nk)
    # select the candiate with the most inliers
    comb_rpje[not_most_inliers] = np.inf
    # break ties by comparing reprojection error
    best_idx = np.argmin(comb_rpje, axis=0) # (nk, )

    ids = [i for i in range(nk)]
    best_candidate = candidates[best_idx, ids, :] # (nk, 3)
    outliers = outliers[best_idx, :, ids].T # (nc, nk)

    # refine 3d locations by gradient method
    p2d_inliers = copy.deepcopy(p2d)
    p2d_inliers[outliers] = np.nan # use only inlier 2d points for refinement
    best_candidate = refine_3d(cameras=cameras, p3d=best_candidate, p2d=p2d_inliers, thresh=thresh)

    return best_candidate, outliers


def refine_3d(cameras: Cameras, p3d: np.ndarray, p2d: np.ndarray, thresh):
    """
    - cameras: Cameras object
    - p3d: (nk, 3), initial 3d locations
    - p2d: (nc, nk, 2)
    """
    # select valid 3d points (seen in at least 2 views)
    valid_3d = np.where(np.isnan(p3d).sum(1) == 0)
    p3d_valid = p3d[valid_3d] # (nj, 3)

    # select valid 2d points (not nan), and stack them into an observation array
    p2d = cameras.undistort_points(p2d) # undistort
    p2d_valid = p2d[:, valid_3d].squeeze() # (nc, nj, 2)
    valid_2d = np.where(np.isnan(p2d_valid).sum(-1) == 0)
    cids = valid_2d[0] # (nob, )
    pids = valid_2d[1] # (nob, )
    p2d_stacked = p2d_valid[valid_2d].reshape(-1, 2) # (nob, 2)

    # compute jacobian matrix for better efficiency
    m = len(cids) * 2
    n = len(p3d_valid) * 3
    A = lil_matrix((m, n), dtype=int)
    ids = np.arange(m // 2)
    for j in range(3):
        A[2*ids, 3*pids+j] = 1
        A[2*ids+1, 3*pids+j] = 1

    # refine 3d locations
    result = least_squares(fun=fun_tri, 
                           x0=p3d_valid.ravel(), 
                           method='trf', 
                           jac_sparsity=A, 
                           x_scale='jac', 
                           args=(cameras.P, p2d_stacked, cids, pids), 
                           verbose=0,
                           loss='soft_l1',
                           f_scale=thresh)
    
    p3d_optm = result.x.reshape(-1, 3) # (nj, 3)
    p3d_refined = copy.deepcopy(p3d)
    p3d_refined[valid_3d] = p3d_optm
    return p3d_refined

def fun_tri(p3d: np.ndarray, projection_mats: np.ndarray, p2d: np.ndarray, cids: np.ndarray, pids: np.ndarray):
    """
    - p3d: (nj*3, ), flattened 3d points
    - projection_mats: (nc, 3, 4), camera projection matrices
    - p2d: (nob, 2), stacked observation 2d points 
    - cids: (nob, ), camera index for each observation
    - pids: (nob, ), 3d point index for each observation 
    """
    nj = len(p3d) // 3
    p3d_h = np.concatenate((p3d.reshape(-1, 3), np.ones((nj, 1))), axis=-1)
    prj2d_h = np.einsum('jkl, jl -> jk', projection_mats[cids], p3d_h[pids])
    prj2d = prj2d_h[:, :2] / prj2d_h[:, 2:3]
    residual = (prj2d - p2d).ravel()
    return residual