import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('TkAgg')


class Cameras:
    def __init__(self, 
                 image_sizes: np.ndarray, 
                 fs: np.ndarray = None, 
                 ms: np.ndarray = None, 
                 cs: np.ndarray = None, 
                 distorts: np.ndarray = None,
                 rvecs: np.ndarray = None, 
                 tvecs: np.ndarray = None):
        """
        - image_sizes: (nc, 2), first w then h, nc refers to 'number of cameras'
        - fs: (nc, ), focal lengths in realworld unit (mm), will be 20. if None
        - ms: (nc, 2), pixel scaling factors, first w then h, will be max(h, w) / fs if None
        - cs: (nc, 2), principle points in pixels, will be [w/2, h/2] if None
        - distorts: (nc, 2), distortion coefficients, will be [0., 0.] if None
        - rvecs: (nc, 3), rotational vectors, will be [0., 0., 0.] if None
        - tvecs: (nc, 3), translational vectors, will be [0., 0., 0.] if None
        """
        self.image_sizes = image_sizes # (w, h)
        self.nc = len(image_sizes)
        self.fs = fs if fs is not None else np.full((self.nc, ), 20.0) # (nc, )
        self.ms = ms if ms is not None else self.image_sizes.max(axis=1)[..., None] / self.fs # (nc, 2)
        self.cs = cs if cs is not None else self.image_sizes / 2 # (nc, 2)
        self.distorts = distorts if distorts is not None else np.zeros((self.nc, 2)) # (nc, 2)
        self.rvecs = rvecs if rvecs is not None else np.zeros((self.nc, 3)) # (nc, 3)
        self.tvecs = tvecs if tvecs is not None else np.zeros((self.nc, 3)) # (nc, 3)

        self.K = compute_K(self.fs, self.ms, self.cs) # (nc, 3, 3), intrinsic matrix
        self.M = compute_M(self.rvecs, self.tvecs) # (nc, 4, 4), extrinsic matrix
        self.P = compute_P(self.K, self.M) # (nc ,3, 4), projection matrix

    def __getitem__(self, indices):
        if type(indices) == int:
            indices = [indices]
        return Cameras(image_sizes=self.image_sizes[indices],
                       fs=self.fs[indices],
                       ms=self.ms[indices],
                       cs=self.cs[indices],
                       distorts=self.distorts[indices],
                       rvecs=self.rvecs[indices],
                       tvecs=self.tvecs[indices])
    
    
    @staticmethod
    def check_valid(length: int, *args):
        for arg in args:
            if arg is not None:
                if arg.shape[0] != length:
                    return False
        return True


    def update_intrinsics(self, 
                          fs: np.ndarray = None, 
                          ms: np.ndarray = None, 
                          cs: np.ndarray = None, 
                          distorts: np.ndarray = None, 
                          indices = None):
        """
        Update camera intrinsics with provided parameters (fs, ms, cs and distorts):
        1. If indices is None, update all cameras, parameters should have length nc
        2. If indices is not None, update specific cameras, parameters should have length ni (number of indices)
        3. Skip updates for parameters set to None
        """
        if indices is None:
            indices = [i for i in range(self.nc)] # update all the cameras
        elif type(indices) == int:
            indices = [indices] # update only 1 camera
            fs = fs[None, ...] if fs is not None else fs
            ms = ms[None, ...] if ms is not None else ms
            cs = cs[None, ...] if cs is not None else cs
            distorts = distorts[None, ...] if distorts is not None else distorts
        ni = len(indices)

        # check dimensional validity
        if not self.check_valid(ni, fs, ms, cs, distorts):
            raise ValueError('Dimension of provided parameters do not match with given indices')

        self.fs[indices] = fs if fs is not None else self.fs[indices]
        self.ms[indices] = ms if ms is not None else self.ms[indices]
        self.cs[indices] = cs if cs is not None else self.cs[indices]
        self.distorts[indices] = distorts if distorts is not None else self.distorts[indices]
        self.K = compute_K(self.fs, self.ms, self.cs)
        self.P = compute_P(self.K, self.M)

    def update_extrinsics(self, rvecs: np.ndarray = None, tvecs: np.ndarray = None, indices = None):
        if indices is None:
            indices = [i for i in range(self.nc)] # update all the cameras
        elif type(indices) == int:
            indices = [indices] # update only 1 camera
            rvecs = rvecs[None, ...] if rvecs is not None else rvecs
            tvecs = tvecs[None, ...] if tvecs is not None else tvecs
        ni = len(indices)

        # check dimensional validity
        if not self.check_valid(ni, rvecs, tvecs):
            raise ValueError('Dimension of provided parameters do not match with given indices')
        
        self.rvecs[indices] = rvecs if rvecs is not None else self.rvecs[indices]
        self.tvecs[indices] = tvecs if tvecs is not None else self.tvecs[indices]
        self.M = compute_M(self.rvecs, self.tvecs) 
        self.P = compute_P(self.K, self.M) 

    def update_M(self, M: np.ndarray, indices = None):
        if indices is None:
            indices = [i for i in range(self.nc)] # update all the cameras
        elif type(indices) == int:
            indices = [indices] # update only 1 camera
            M = M[None, ...]
        ni = len(indices)

        # check dimensional validity
        if not self.check_valid(ni, M):
            raise ValueError('Dimension of provided parameters do not match with given indices')
        
        self.rvecs[indices] = Rotation.from_matrix(M[:, :3, :3]).as_rotvec()
        self.tvecs[indices] = M[:, :3, 3]
        self.M[indices] = M
        self.P = compute_P(self.K, self.M)

    def scale(self, new_sizes: np.ndarray = None, world_scale: float = None):
        """
        Adjust the camera parameters to ensure that the projection remains consistent 
        under changes to the image resolution or scaling of the world frame.

        - new_sizes: (nc, 2) array, change to new image size
        - world_scale: float, scale the world and the camera with a factor world_scale
        """
        # image resize
        if new_sizes is not None:
            image_scale = new_sizes / self.image_sizes # (nc, 2)
            sx, sy = image_scale[:, 0], image_scale[:, 1] # (nc, )
            # scale mx, my
            new_ms = np.zeros_like(self.ms) # (nc, 2)
            new_ms[:, 0] = self.ms[:, 0] * sx
            new_ms[:, 1] = self.ms[:, 1] * sy
            # scale cx, cy
            new_cs = np.zeros_like(self.cs) # (nc, 2)
            new_cs[:, 0] = self.cs[:, 0] * sx
            new_cs[:, 1] = self.cs[:, 1] * sy
            self.update_intrinsics(ms=new_ms, cs=new_cs)
            self.image_sizes = new_sizes

        # world resize
        if world_scale is not None:
            new_tvecs = self.tvecs * world_scale
            self.update_extrinsics(tvecs=new_tvecs)


    # faster than opencv projection
    def project_points(self, p3d: np.ndarray):
        # p3d: (nk, 3)
        # world to camera
        p3d_h = np.concatenate((p3d, np.ones((p3d.shape[0], 1))), axis=-1) # (nk, 4)
        p3d_c = np.einsum('ijk, lk -> ilj', self.M, p3d_h) # (nc, 4, 4) @ (nk, 4) -> (nc, nk, 4)
        # distortion
        x, y = p3d_c[..., 0] / p3d_c[..., 2], p3d_c[..., 1] / p3d_c[..., 2] # (nc, nk)
        r2 = x ** 2 + y ** 2 # (nc, nk)
        dist = 1 + self.distorts[:, 0, None] * r2 + self.distorts[:, 1, None] * (r2 ** 2) # (nc, nk)
        xd, yd = x * dist, y * dist # (nc, nk)
        # camera to image
        u, v = xd * self.K[:, 0, 0, None] + self.K[:, 0, 2, None], yd * self.K[:, 1, 1, None] + self.K[:, 1, 2, None] # (nc, nk)
        p2d = np.stack([u, v], axis=-1) # (nc, nk, 2)

        return p2d # (nc, nk, 2)

    def undistort_points(self, p2d: np.ndarray):
        # p2d: (nc, nk, 2)
        new_p2d = []
        for i in range(self.nc):
            p2d_sub = copy.deepcopy(p2d[i]) # (nk, 2)
            valid = np.isnan(p2d_sub).sum(1) == 0
            dist_sub = np.hstack([self.distorts[i], np.zeros(2)]) # (4, )
            undist = cv2.undistortImagePoints(src=p2d_sub[valid], cameraMatrix=self.K[i], distCoeffs=dist_sub) # (nk_valid, 1, 2)
            p2d_sub[valid] = undist.squeeze() # (nk, 2)
            new_p2d.append(p2d_sub)

        return np.stack(new_p2d) # (nc, nk, 2)
    

    def plot(self, points: np.ndarray = None, yz_flip=False, factor=0.05, margin=1.1, psize=0.1):
        """
        - factor: float, to control how big the cameras look like in the world frame
        - points: (nk, 3), if not None, plot given 3d points together with cameras
        """
        camera_poses = np.linalg.inv(self.M)
        positions = camera_poses[:, :3, 3] # (nc, 3)
        center = positions.mean(axis=0) # (3, )
        # determine the center of the plot
        if points is None:
            all_points = positions
        else:
            valid = np.isnan(points).sum(1) == 0
            points = points[valid]
            # center = points.mean(axis=0)
            all_points = np.vstack([positions, points])

        nk = all_points.shape[0]
        dists = pair_euclidean_nan(all_points, np.repeat(center[None, ...], nk, axis=0), mean=False) # (np, )
        expand = np.nanmax(dists) * margin
        expand = expand if expand != 0 else 1. # in case where there is only 1 camera
        pscale = expand * factor

        cam_points0 = np.array([[0., 0., 0.],
                            [2, 1.5, 3.],
                            [-2, 1.5, 3.],
                            [2, -1.5, 3.],
                            [-2, -1.5, 3.]]) * pscale
        
        axis_points0 = np.array([[0., 0., 0.],
                                [2., 0., 0.],
                                [0., 2., 0.],
                                [0., 0., 2.]]) * pscale
        
        text_point0 = np.array([[0., 0., -3.]]) * pscale
        
        num_cam = len(camera_poses)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for j in range(num_cam):
            R, t = camera_poses[j, :3, :3], camera_poses[j, :3, 3]
            cam_points = (R @ cam_points0.T + t[:, None]).T
            axis_points = (R @ axis_points0.T + t[:, None]).T
            text_point = (R @ text_point0.T + t[:, None]).T
            # plot a plane
            plane = cam_points[[1, 2, 4, 3, 1]]
            if yz_flip:
                ax.plot(plane[:, 0], plane[:, 2], plane[:, 1], c='red', alpha=0.6)
            else:
                ax.plot(plane[:, 0], plane[:, 1], plane[:, 2], c='red', alpha=0.6)
            for i in range(4):
                line_points = cam_points[[0, i+1]]
                if yz_flip:
                    ax.plot(line_points[:, 0], line_points[:, 2], line_points[:, 1], c='red', alpha=0.6)
                else:
                    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], c='red', alpha=0.6)

            # connect the plane with the camera center
            for i in range(3):
                line_points = axis_points[[0, i+1]]
                if yz_flip:
                    ax.plot(line_points[:, 0], line_points[:, 2], line_points[:, 1], c='dodgerblue')
                else:
                    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], c='dodgerblue')
            # add text
            if yz_flip:
                ax.text(text_point[:, 0][0], text_point[:, 2][0], text_point[:, 1][0], f"Cam {j}", color='black')
            else:
                ax.text(text_point[:, 0][0], text_point[:, 1][0], text_point[:, 2][0], f"Cam {j}", color='black')

        if points is not None:
            if yz_flip:
                ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=psize, c='blue')
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=psize, c='blue')

        # canvas setting
        ax.set_title('Cameras')
        ax.grid(linestyle='--', visible=True)
        ax.set_xlabel('X')
        ax.set_xlim((center[0] - expand, center[0] + expand))
        if yz_flip:
            ax.set_zlabel('Y')
            ax.set_ylabel('Z')
            ax.set_zlim((center[1] - expand, center[1] + expand))
            ax.set_ylim((center[2] - expand, center[2] + expand))
        else:
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_ylim((center[1] - expand, center[1] + expand))
            ax.set_zlim((center[2] - expand, center[2] + expand))

        # show
        plt.show()


def compute_K(fs: np.ndarray, ms: np.ndarray, cs: np.ndarray):
    """
    - fs: (nc, )
    - ms: (nc, 2)
    - cs: (nc, 2)
    """
    nc = len(fs)
    K = np.zeros((nc, 3, 3))
    K[:, 0, 0] = fs * ms[:, 0]
    K[:, 1, 1] = fs * ms[:, 1]
    K[:, 0, 2] = cs[:, 0]
    K[:, 1, 2] = cs[:, 1]
    K[:, 2, 2] = 1
    return K # (nc, 3, 3)


def compute_M(rvecs: np.ndarray, tvecs: np.ndarray):
    """
    - rvecs: (nc, 3), rotational vectors, will be [0., 0., 0.] if None
    - tvecs: (nc, 3), translational vectors, will be [0., 0., 0.] if None
    """
    nc = len(rvecs)
    M = np.zeros((nc, 4, 4))
    M[:, :3, :3] = Rotation.from_rotvec(rvecs).as_matrix()
    M[:, :3, 3] = tvecs
    M[:, 3, 3] = 1
    return M # (nc, 4, 4)

def compute_P(K: np.ndarray, M: np.ndarray):
    """
    - K: (nc, 3, 3)
    - M: (nc, 4, 4)
    """
    nc = len(K)
    K_extend = np.concatenate((K, np.zeros((nc, 3, 1))), axis=-1)
    P = K_extend @ M
    return P # (nc, 3, 4)

def pair_euclidean_nan(x1, x2, mean=True):
    if mean:
        return np.nanmean(((x1 - x2)**2).sum(-1)**0.5)
    else:
        return ((x1 - x2)**2).sum(-1)**0.5