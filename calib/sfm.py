import numpy as np
import cv2
import copy
from cameras import Cameras, compute_K, pair_euclidean_nan
from triangulation import triangulate, triangulate_robust
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from itertools import combinations
from typing import Union, Callable


def p2d_overlap(p2d0: np.ndarray, p2d1: np.ndarray):
    """
    - p2d0, p2d1: (nj, 2) 2d points, np.nan if unseen
    """
    visible0 = np.isnan(p2d0).sum(1) == 0
    visible1 = np.isnan(p2d1).sum(1) == 0
    id_overlap = np.where(visible0 & visible1)
    return p2d0[id_overlap], p2d1[id_overlap], id_overlap


def estimate_pose_epipolar(p2d0: np.ndarray, 
                           p2d1: np.ndarray, 
                           K0: np.ndarray, 
                           K1: np.ndarray, 
                           distort1: np.ndarray, 
                           distort2: np.ndarray):
    """
    - p2d0, p2d1: (nj, 2) 2d points, can contain nans 
    - K0, K1: (3, 3), intrinsic matrix
    - distort1, distort2: (2, ), distortion 
    """
    # select points that are seen in both views
    p2d0, p2d1, _ = p2d_overlap(p2d0, p2d1)
    
    # distortion
    add_zeros = np.zeros(2)
    distort1 = np.hstack([distort1, add_zeros])
    distort2 = np.hstack([distort2, add_zeros])

    # estimate pose using epipolar geometry
    _, _, R, t, _ = cv2.recoverPose(cameraMatrix1=K0, 
                                    cameraMatrix2=K1, 
                                    points1=p2d0, 
                                    points2=p2d1, 
                                    distCoeffs1=distort1, 
                                    distCoeffs2=distort2)
    relative_trans = np.eye(4, 4)
    relative_trans[:3, :3] = R
    relative_trans[:3, 3] = t.squeeze() * 10
    return relative_trans # (4, 4)


def estimate_pose_pnp(p3d: np.ndarray, p2d: np.ndarray, K: np.ndarray, distort: np.ndarray, thresh: float = 5.0):
    """
    - p3d: (nj, 3), points in 3d, can have nans
    - p2d: (nj, 2), points in 2d, can have nans
    - K: (3, 3), camera intrinsic
    - thresh: float, reprojection error below which points are considered as inliers in the ransac pnp solver
    - n_iter: int, number of iterations in the solver
    """
    # find points that are seen in both 3d and 2d
    valid_3d = np.isnan(p3d).sum(1) == 0
    valid_2d = np.isnan(p2d).sum(1) == 0
    valid = np.where(valid_3d & valid_2d)
    p3d, p2d = p3d[valid], p2d[valid]

    # find camera pose using ransac pnp solver
    add_zeros = np.zeros(2)
    distort = np.hstack([distort, add_zeros])
    trans = np.eye(4, 4)
    _, rvec, t, _ = cv2.solvePnPRansac(objectPoints=p3d, imagePoints=p2d, cameraMatrix=K, distCoeffs=distort, reprojectionError=thresh)
    R = Rotation.from_rotvec(rvec.squeeze()).as_matrix()
    trans[:3, :3] = R
    trans[:3, 3] = t.squeeze()
    return trans # (4, 4)



class SFM:
    def __init__(self, image_sizes: np.ndarray, fs: np.ndarray, ms: np.ndarray, cs: np.ndarray, distorts: np.ndarray, p2d: np.ndarray):
        """
        - image_sizes: (nc, 2), image sizes for cameras
        - K: (nc, 3, 3), intrinsic matrices
        - p2d: (nc, nk, 2), 2d points
        """
        self.cameras: Cameras = Cameras(image_sizes=image_sizes, fs=fs, ms=ms, cs=cs, distorts=distorts)
        self.p2d = p2d # (nc, nk, 2)
        self.nk = p2d.shape[1]
        self.visible = np.isnan(p2d).sum(2) == 0 # (nc, nk), whether point k is visible in view c
        self.outliers = np.full((self.cameras.nc, self.nk), False) # (nc, nk), outlier 2d points
        self.p3d = np.full((self.nk, 3), np.nan) # (nk, 3)
        self.known_3d = np.full((self.nk, ), False) # (nk, ), whether point k has know 3d location (has been triangulated)
        self.views_in = [] # views that have been registered
        self.views_out = [i for i in range(self.cameras.nc)] # views that haven't been registered


    def update_p3d(self, p3d: np.ndarray):
        if p3d.shape[0] != self.nk:
            raise ValueError('Number of points not consistent')
        self.p3d = p3d
        self.known_3d = np.isnan(p3d).sum(1) == 0


    def initialize(self):
        """
        Initialize with two cameras that have the largest overlap
        - threshold: float, used in ransac epipolar geometry method
        - verbose: whether print information
        """
        if len(self.views_in) != 0:
            raise ValueError('Duplicated Initialization!')
        
        combs = list(combinations(self.views_out, 2)) # all possible combinations for two views
        count: list = []
        for comb in combs:
            overlap = self.visible[comb[0]] & self.visible[comb[1]]
            count.append(overlap.sum()) # number of points seen in both views

        best_comb = combs[count.index(max(count))] 
        id0, id1 = best_comb
        # estimate the transformation of camera[id1] with respect to camera[id0] using epipolar geometry
        # the estimated translation is up to scale
        rela_trans = estimate_pose_epipolar(p2d0=self.p2d[id0], 
                                            p2d1=self.p2d[id1], 
                                            K0=self.cameras.K[id0], 
                                            K1=self.cameras.K[id1],
                                            distort1=self.cameras.distorts[id0],
                                            distort2=self.cameras.distorts[id1]) # (4, 4)

        self.cameras.update_M(M=rela_trans, indices=id1) # update cameras
        self.views_in += [id0, id1]
        self.views_out.remove(id0)
        self.views_out.remove(id1)

        # triangulation to get 3d points
        # sub_cam = self.cameras[self.views_in]
        # p3d = triangulate(cameras=sub_cam, p2d=self.p2d[self.views_in]) # p3d: (nk, 3), outliers: (nc_sub, nk)
        # rpj = sub_cam.project_points(p3d) # (nc_sub, nk, 2)
        # rpje = pair_euclidean_nan(rpj, self.p2d[self.views_in]) # (nc_sub, nk)
        # self.update_p3d(p3d)

        # print information of initialization
        print('-'*100)
        print('Initialization: \n')
        print(f'* Camera [{id0}] and [{id1}] have the largest overlap, with {max(count)} points seen in both \n')
        print(f'* Relative rotation: \n {rela_trans[:3, :3]} \n')
        print(f'* Relative translation (up to scale): \n {rela_trans[:3, 3]} \n')
        print(f'* Cameras registered: [{len(self.views_in)}/{self.cameras.nc}]')
        print('-'*100)
        

    def register_new(self, thresh=5.0):
        """
        Select the view that sees the most triangulated 3d points, estimate its pose by solving PnP

        - thresh: float, threshold for ransac pnp solver
        """
        if len(self.views_out) == 0:
            raise ValueError('All cameras have been registered')
        
        overlap = self.known_3d & self.visible[self.views_out]
        best_view = self.views_out[np.argmax(overlap.sum(1))]
        # estimate pose with pnp
        trans = estimate_pose_pnp(p3d=self.p3d, 
                                  p2d=self.p2d[best_view], 
                                  K=self.cameras.K[best_view], 
                                  distort=self.cameras.distorts[best_view],
                                  thresh=thresh) # (4, 4)

        self.cameras.update_M(M=trans, indices=best_view)
        self.views_in.append(best_view)
        self.views_out.remove(best_view)

        # print
        print('-'*100)
        print(f'New View Registeration: \n')
        print(f'* Camera [{best_view}] is registered, with [{overlap.sum(1).max()}/{self.known_3d.sum()}] points seen \n')
        # print(f'* [{id_new3d.sum()}] new points are triangulated, now there are [{self.known_3d.sum()}] 3D points in total \n')
        print(f'* Cameras registered: [{len(self.views_in)}/{self.cameras.nc}]')
        print('-'*100)


    def triangulate_points(self, views=None, robust=True, thresh=10.0, max_samples=None, verbose=True):
        """
        - views: list[int], if not specified all registered views will be used
        - robust: bool, whether to perform robust triangulation, if False will do svd triangulation
        - thresh: float, threshold for outliers in robust triangulation
        - max_samples: used in robust triangulation
        - verbose: ...
        """
        views = views if views is not None else self.views_in
        sub_cam = self.cameras[views]
        if robust:
            p3d, outliers = triangulate_robust(cameras=sub_cam, p2d=self.p2d[views], thresh=thresh, max_samples=max_samples)
            self.outliers[views] = outliers # (nc_sub, nk)
        else:
            p3d = triangulate(cameras=sub_cam, p2d=self.p2d[views])
        rpj = sub_cam.project_points(p3d) # (nc_sub, nk, 2)
        rpje = pair_euclidean_nan(rpj, self.p2d[views]) # (nc_sub, nk)
        self.update_p3d(p3d)

        rpje_in = pair_euclidean_nan(rpj, self.p2d[views], mean=False)
        rpje_in[rpje_in > thresh] = np.nan
        rpje_in_mean = np.nanmean(rpje_in)
        
        if verbose:
            print('-'*100)
            print(f'Triangulation with cameras {views}: \n')
            print(f'* Number of triangulated points: [{self.known_3d.sum()}]')
            print(f'* Average reprojection error for triangulated points: [{rpje: .2f} pixels] \n')
            print(f'                                                      [{rpje_in_mean: .2f} pixels] (inliers only) \n')
            print('-'*100)


    def bundle_adjust(self, views=None, update_3d=True, update_f=False, update_distort=False, inliers_only=False, loss_func='soft_l1', thresh=10.0, n_iter=200, verbose=True):
        """
        Perform sparse bundle adjustment on given views (simultaneously refine 3d points and camera parameters)

        - views: the views on which ba will be perfomed
        - update_3d, bool, whether to optimize 3d locations
        - update_f: bool, whether to optimize focal lengths of cameras
        - update_distort: bool, whether to optimize distortion parameters
        - inliers_only: bool, use all p2ds or only inliers for ba
        - loss_func: str or callable, loss function used in ba, is a robust loss (huber) by default
        - thresh: float, the threshold of reprojection error used in robust loss function
        - n_iter: int, iterations for the ba solver
        - verbose: ...
        """
        # compute reprojection error before BA
        if views is None:
            views = self.views_in # global ba (on all registered views)
        elif type(views) == int:
            views = [views]
            
        sub_cam = copy.deepcopy(self.cameras[views])
        rpj = sub_cam.project_points(self.p3d) # (nc, nk, 2)
        rpje = pair_euclidean_nan(rpj, self.p2d[views], mean=False) # (nc, nk)
        sorted_rpje = np.sort(rpje[np.where(~np.isnan(rpje))])

        p2d = copy.deepcopy(self.p2d[views])
        if inliers_only:
            p2d[self.outliers[views]] = np.nan # ingnore outliers
            rpje_in = pair_euclidean_nan(rpj, p2d, mean=False) # (nc, nk)
            sorted_rpje_in = np.sort(rpje_in[np.where(~np.isnan(rpje_in))])


        # bundle adjustment
        print(f'Start Bundle Adjustment: \n')
        new_cams, new_p3d = ba(cameras=sub_cam, 
                               p3d=self.p3d, 
                               p2d=p2d,
                               update_3d=update_3d,
                               update_f=update_f,
                               update_distort=update_distort, 
                               verbose=2 if verbose else 1, 
                               loss_func=loss_func, 
                               thresh=thresh, 
                               nfev=n_iter)
        
        # compute rpje after BA
        rpj_new = new_cams.project_points(new_p3d) # (nc, nk, 2)
        rpje_new = pair_euclidean_nan(rpj_new, self.p2d[views], mean=False) # (nc, nk)
        sorted_rpje_new = np.sort(rpje_new[np.where(~np.isnan(rpje_new))])
        if inliers_only:
            rpje_new_in = pair_euclidean_nan(rpj_new, p2d, mean=False) # (nc, nk)
            sorted_rpje_new_in = np.sort(rpje_new_in[np.where(~np.isnan(rpje_new_in))])

        # update camera parameters
        self.cameras.update_M(M=new_cams.M, indices=views)
        self.cameras.update_intrinsics(fs=new_cams.fs, distorts=new_cams.distorts, indices=views)
        self.update_p3d(new_p3d) 
        

        if verbose:
            print('-'*100)
            print(f'Bundle adjustment: \n')
            print(f'* Cameras adjusted: {views} \n')
            
            print(f'* Reprojection error before BA: - Mean: [{np.mean(sorted_rpje): .2f} pixels], - Q3: [{np.percentile(sorted_rpje, 75): .2f} pixels] \n') 
            if inliers_only:
                print(f'                                - Mean: [{np.mean(sorted_rpje_in): .2f} pixels], - Q3: [{np.percentile(sorted_rpje_in, 75): .2f} pixels] (inliers only) \n')

            print(f'* Reprojection error after  BA: - Mean: [{np.mean(sorted_rpje_new): .2f} pixels], - Q3: [{np.percentile(sorted_rpje_new, 75): .2f} pixels] \n')
            if inliers_only:
                print(f'                                - Mean: [{np.mean(sorted_rpje_new_in): .2f} pixels], - Q3: [{np.percentile(sorted_rpje_new_in, 75): .2f} pixels] (inliers only) \n')

            print('-'*100)


    def calibrate(self, 
                  rpje_max: float = 10.0,
                  ba_step: int = 3,
                  ba_loss: Union[str, Callable] ='soft_l1',
                  update_f: bool = False, 
                  update_distort: bool = False, 
                  ):
        """
        Introduction ...

        - rg_thresh: outlier threshold for registering new views,
        - ta_thresh: threshold for robust triangulation (only effective when ta_rb=True),
        - ba_interval: the number of new cameras to register before performing bundle adjustment, 
        - ba_automatic: whether to perform automatical threshold downgrading ba, if true:
                        before performing bundle adjustment, compute the rpje and record the 75%, 50%, 25%, and 15% percentile errors.
                        perform ba for 4 times, each time using the corresponding error (from 75% to 15%) as the outlier threshold, gradually lowering it.
        - ba_loss: loss function for ba, 
                        can scipy supported functions (str: 'linear', 'soft_l1', 'huber', 'cauchy', 'arctan') or a self implemented callable
        - ba_niter: how many iterations to run in each global ba, if None iteration will end by meeting least_square stopping criterions (xtol, gtol ...)
        - ba_niter_local: how many iterations to run in each local ba
        - update_f: whether to optimize focal length in ba,
        - update_distort: whether to optimize distortion coefficients in ba,
        - inliers_only: only use inliers for ba,
        - ba_thresh: (only effective when ba_automatic=False), the manually choosen outlier threshold for ba,
        - verbose: print log,
        """

        # Initialization
        self.initialize()
        self.triangulate_points(robust=False, verbose=True)
        print('\n \n \n') 
        # Register new views
        while len(self.views_out):
            # register new view by solving pnp
            self.register_new()
            # local ba: refine the latest added view
            self.bundle_adjust(views=self.views_in[-1], 
                               update_3d=False, # 3d points fixed
                               update_f=update_f, 
                               update_distort=update_distort, 
                               inliers_only=False, 
                               loss_func=ba_loss, 
                               n_iter=50, 
                               thresh=rpje_max,
                               verbose=False)
            # re-triangulation
            self.triangulate_points(robust=True, thresh=rpje_max, max_samples=20, verbose=False)
            # global ba: refine all registered views 
            if len(self.views_in) % ba_step == 0:
                rpj = self.cameras[self.views_in].project_points(self.p3d)
                rpje = pair_euclidean_nan(rpj, self.p2d[self.views_in], mean=False) # (nc, nk)
                sorted_rpje = np.sort(rpje[np.where(~np.isnan(rpje))]) # (nc*nk, )
                rpje_low = np.percentile(sorted_rpje, 25)
                ba_threshes = np.linspace(rpje_max, rpje_low, num=4)
                # Iteratively donwgrade error threshold
                for thresh in ba_threshes:
                    self.bundle_adjust(update_f=update_f, 
                                       update_distort=update_distort, 
                                       inliers_only=True, 
                                       loss_func=ba_loss, 
                                       n_iter=100,
                                       thresh=thresh)
            print('\n \n \n') 
                    
        # BA after all views are registered    
        self.bundle_adjust(update_f=update_f, 
                           update_distort=update_distort, 
                           inliers_only=True, 
                           loss_func=ba_loss, 
                           n_iter=200,
                           thresh=ba_threshes[-1])



def ba(cameras: Cameras, 
       p3d: np.ndarray, 
       p2d: np.ndarray,
       update_3d: bool, 
       update_f: bool, 
       update_distort: bool, 
       verbose: int, 
       loss_func: Union[str, Callable], 
       thresh: float, 
       nfev: int) -> tuple[Cameras, np.ndarray]:
    ...

    # collect camera parameters to be optimized
    rvecs = cameras.rvecs # (nc, 3)
    tvecs = cameras.tvecs # (nc, 3)
    cam_parameters = np.hstack([rvecs, tvecs]) # (nc, 6)
    if update_f:
        cam_parameters = np.hstack([cam_parameters, cameras.fs[..., None]]) # (nc, 6+1) = (nc, 7)
    if update_distort:
        cam_parameters = np.hstack([cam_parameters, cameras.distorts]) # (nc, 7+2) = (nc, 9)
    ncp = cam_parameters.shape[1]
    cam_parameters = cam_parameters.ravel() # (nc*ncp, )

    # select valid 3d points (seen in at least 2 views)
    valid_3d = np.isnan(p3d).sum(1) == 0
    p3d_valid = p3d[valid_3d].ravel() # (nj*3, )

    # select valid 2d points (not nan), and stack them into an observation array
    p2d_valid = p2d[:, valid_3d] # (nc, nj, 2)
    valid_2d = np.where(np.isnan(p2d_valid).sum(-1) == 0)
    cids = valid_2d[0] # (nob, )
    pids = valid_2d[1] # (nob, )
    p2d_stacked = p2d_valid[valid_2d].reshape(-1, 2) # (nob, 2)

    # construct jacobian matrix for better efficiency
    if update_3d:
        x0 = np.concatenate([cam_parameters, p3d_valid]) # (nc*ncp + nj*3, )
    else:
        x0 = cam_parameters # (nc*ncp, ), not to update 3d points
    m = len(p2d_stacked) * 2
    n = len(x0)
    A = lil_matrix((m, n), dtype=int) # (nob*2, nc*ncp + nj*3) or (nob*2, nc*ncp)
    ids = np.arange(m // 2)
    # where camera parameters have grad
    for s in range(ncp):
        A[2*ids, cids*ncp + s] = 1
        A[2*ids + 1, cids*ncp + s] = 1
    # where 3d points have grad
    if update_3d:
        for s in range(3):
            A[2*ids, cameras.nc*ncp + pids*3 + s] = 1
            A[2*ids + 1, cameras.nc*ncp + pids*3 + s] = 1

    # sparse bundle adjustment
    points3d = None if update_3d else p3d_valid.reshape(-1, 3)
    fs = None if update_f else cameras.fs
    distorts = None if update_distort else cameras.distorts
    args = (cameras.ms, cameras.cs, points3d, fs, distorts, p2d_stacked, cids, pids)
    result = least_squares(fun=func_ba, 
                           x0=x0, 
                           method='trf', 
                           x_scale='jac',
                           jac_sparsity=A, 
                           args=args, 
                           verbose=verbose, 
                           loss=loss_func,
                           f_scale=thresh,
                           max_nfev=nfev,
                           ftol=None if nfev is not None else 1e-8,
                           xtol=None if nfev is not None else 1e-8,)
    
    # optimized camera parameters
    new_cameras = copy.deepcopy(cameras)
    params_optm = result.x[: cameras.nc*ncp].reshape(-1, ncp) # (nc, ncp)
    extrinsics_optm = params_optm[:, :6] # (nc, 6)
    new_cameras.update_extrinsics(rvecs=extrinsics_optm[:, :3], tvecs=extrinsics_optm[:, 3:]) # extrinsics
    if update_f:
        fs_optm = params_optm[:, 6] # (nc, )
        new_cameras.update_intrinsics(fs=fs_optm)
    if update_distort:
        distort_optm = params_optm[:, 7:] # (nc, 2)
        new_cameras.update_intrinsics(distorts=distort_optm)
    
    # optimized 3d locations
    if update_3d:
        p3d_optm = result.x[cameras.nc*ncp: ].reshape(-1, 3) # (nj, 3)
        p3d_final = copy.deepcopy(p3d)
        p3d_final[valid_3d] = p3d_optm
    else:
        p3d_final = copy.deepcopy(p3d)

    return new_cameras, p3d_final
    


def func_ba(x0: np.ndarray, ms: np.ndarray, cs: np.ndarray, p3d: np.ndarray, fs: np.ndarray, distorts: np.ndarray, p2d: np.ndarray, cids: np.ndarray, pids: np.ndarray):
    """
    - x0: (nc*ncp + nj*3, ), raveled camera parameters and 3d locations to be optimized
    - ms: (nc, 2), camera pixel scaling factors
    - cs: (nc, 2), camera principle points
    - fs: (nc, ) if not optimize focal length else None
    - distorts: (nc, 2) if not optimize distortion parameters else None
    - p2d: (nob, 2), 2d observations
    - cids: (nob, ), camera index for each 2d observation
    - pids: (nob, ), 3d point index for each 2d observation
    """
    nc = len(ms)
    ncp = 6
    if fs is None: # if focal length is to be optimized
        ncp += 1
    if distorts is None: # if distortion is to be optimized
        ncp += 2

    cam_params = x0[: nc*ncp].reshape(nc, ncp) # (nc, ncp)
    if p3d is None:
        p3d = x0[nc*ncp: ].reshape(-1, 3) # (nj, 3)
    rvecs, tvecs = cam_params[:, :3], cam_params[:, 3:6] # (nc, 3)
    fs = fs if fs is not None else cam_params[:, 6] # (nc, )
    distorts = distorts if distorts is not None else cam_params[:, 7:] # (nc, 2)
    Ks = compute_K(fs, ms, cs) # (nc, 3, 3)
    rpj = project(rvecs[cids], tvecs[cids], Ks[cids], distorts[cids], p3d[pids]) # (nob, 2)
    residual = (p2d - rpj).ravel() # (nob*2, )
    return residual


def project(rvecs: np.ndarray, tvecs: np.ndarray, Ks: np.ndarray, distorts: np.ndarray, p3d: np.ndarray):
    """
    - rvecs: (nob, 3)
    - tvecs: (nob, 3)
    - Ks: (nob, 3, 3)
    - distorts: (nob, 2)
    - p3d: (nob, 3)
    """
    p3d_c = rotate(p3d, rvecs) + tvecs # (nob, 3)
    x, y = p3d_c[:, 0] / p3d_c[:, 2], p3d_c[:, 1] / p3d_c[:, 2] # (nob, )
    r2 = x ** 2 + y ** 2 # (nob, )
    dist = 1 + distorts[:, 0] * r2 + distorts[:, 1] * (r2 ** 2) # (nob, )
    xd, yd = x * dist, y * dist # (nob, )
    u, v = xd * Ks[:, 0, 0] + Ks[:, 0, 2], yd * Ks[:, 1, 1] + Ks[:, 1, 2] # (nob, )
    prj = np.stack([u, v], axis=-1) # (nob, 2)
    return prj


def rotate(p3d: np.ndarray, rvecs: np.ndarray):
    """
    - points: (nj, 3)
    - rvecs: (nj, 3)
    """
    eps = 1e-6
    theta = np.linalg.norm(rvecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rvecs / (theta + eps)
        v = np.nan_to_num(v)
    dot = np.sum(p3d * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated = cos_theta * p3d + sin_theta * np.cross(v, p3d) + dot * (1 - cos_theta) * v

    return rotated # (nj, 3)