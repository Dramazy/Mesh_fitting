import torch
import time
import os
import numpy as np
import torchvision.transforms.functional as TF
from pytorch3d.utils import cameras_from_opencv_projection
from PIL import Image
from calib.cameras import Cameras



def load_cameras(path):
    cam_data = np.load(path, allow_pickle=True)
    rvecs, tvecs, fs, ms, cs, distorts, image_sizes = [], [], [], [], [], [], []
    for cam in cam_data:
        rvecs.append(cam['rvec'])
        tvecs.append(cam['tvec'])
        fs.append(cam['f'])
        ms.append(cam['m'])
        cs.append(cam['c'])
        distorts.append(cam['distort'])
        image_sizes.append(cam['image_size'])

    cameras = Cameras(image_sizes=np.stack(image_sizes), 
                    fs=np.stack(fs), 
                    ms=np.stack(ms), 
                    cs=np.stack(cs), 
                    distorts=np.stack(distorts), 
                    rvecs=np.stack(rvecs), 
                    tvecs=np.stack(tvecs))
    
    return cameras


def save_cameras(cameras: Cameras, path: str):
    ...


def benchmark_forward(model, input_tensor=None, num_iters=100):
    """
    Measure the speed of a PyTorch model's forward pass in FPS.
    
    Args:
        model: PyTorch model (nn.Module).
        input_tensor: Example input tensor for the forward pass.
        num_iters: Number of iterations to average the speed.
    
    Returns:
        FPS (frames per second).
    """
    # Ensure model is in eval mode and on the correct device
    model.eval()

    # Warm-up GPU (avoids cold start effects)
    with torch.no_grad():
        for _ in range(10):
            if input_tensor is not None:
                model(input_tensor)
            else:
                model()

    # Measure execution time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iters):
            if input_tensor is not None:
                model(input_tensor)
            else:
                model()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    # Compute FPS
    total_time = end_time - start_time
    fps = num_iters / total_time
    return fps


def procrustes(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    return scale, R, t



def convert_cameras(cameras: Cameras):
    """
    convert a Cameras object to a pytoch3d perspective camera
    """
    # calib cameras use (w, h) but pytrorch3d cameras use (h, w)
    hw = torch.FloatTensor(cameras.image_sizes).flip(-1)
    return cameras_from_opencv_projection(R=torch.FloatTensor(cameras.M[:, :3, :3]), 
                                          tvec=torch.FloatTensor(cameras.tvecs), 
                                          camera_matrix=torch.FloatTensor(cameras.K), 
                                          image_size=hw)


def pair_euclidean_nan(x1, x2, mean=True):
    if mean:
        return torch.nanmean((x1 - x2).pow(2).sum(-1).sqrt())
    else:
        return (x1 - x2).pow(2).sum(-1).sqrt()
    

def invert_binary_masks(input_folder: str, output_folder: str) -> None:
    """
    Inverts all binary masks (0->1, 1->0) in a folder and saves them to the output folder.

    Args:
        input_folder (str): Path to the folder containing binary masks.
        output_folder (str): Path to save the inverted masks.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust as needed
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Load image as grayscale
            mask = Image.open(input_path).convert("L")
            mask_tensor = TF.to_tensor(mask)  # Convert to tensor (range 0 to 1)
            
            # Invert mask (0 -> 1, 1 -> 0)
            inverted_mask = 1 - mask_tensor
            
            # Convert back to PIL image and save
            inverted_image = TF.to_pil_image(inverted_mask)
            inverted_image.save(output_path)
            
            print(f"Processed: {filename}")

    print("Inversion complete!")

