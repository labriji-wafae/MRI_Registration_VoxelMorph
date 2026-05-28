import os
import random
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from voxelmorph.tf.losses import NCC, Grad
from tqdm import tqdm
from scipy.ndimage import zoom
from collections import defaultdict
import mlflow
import mlflow.tensorflow
from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects, binary_closing, ball

from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score

from scipy.ndimage import gaussian_filter



def load_and_preprocess_nifti(path, target_shape):
   img = nib.load(path).get_fdata()
#    img = np.transpose(img, (2, 1, 0)) # a verifier Faux
   min_val = img.min()
   max_val = img.max()
   if max_val - min_val == 0:
      return np.zeros_like(img)  # Prevent division by zero
   img = (img - min_val) / (max_val - min_val)

   img = resize_if_needed(img, target_shape)
   img = np.expand_dims(img, axis=(0, -1))  # Shape: [1, H, W, D, 1]
   return img.astype(np.float32)


def resize_if_needed(image, target_shape):
   if image.shape != target_shape:
      from scipy.ndimage import zoom
      zoom_factors = [t / s for t, s in zip(target_shape, image.shape)]
      image = zoom(image, zoom_factors, order=1)
   return image


def load_cavity_mask(path, target_shape, cavity_label=4):
    seg = nib.load(path).get_fdata()
    mask = (seg == cavity_label).astype(np.float32)
    if mask.shape != target_shape :
        zoom_factors = [t/s for t,s in zip(target_shape, mask.shape)]
        mask = zoom(mask, zoom_factors, order=0)
    
    return np.expand_dims(mask, axis=(0, -1))


def load_all_tumor_mask(path, target_shape):
    seg = nib.load(path).get_fdata()
    mask = (seg > 0).astype(np.float32)
    
    if mask.shape != target_shape:
        zoom_factors = [t / s for t, s in zip(target_shape, mask.shape)]
        mask = zoom(mask, zoom_factors, order=0)  # nearest-neighbor interpolation

    return np.expand_dims(mask, axis=(0, -1))


def CSF_Kmeans_segmentor(data):
    brain_m = (data>0).astype(bool)
    voxels = data[brain_m].reshape(-1,1)

    # K-means
    km = KMeans(n_clusters=3, random_state=0).fit(voxels)
    labels = np.zeros(data.shape, int)
    labels[brain_m] = km.labels_ + 1  # labels ∈ {1,2,3}

    # 4) find CSF label (lowest cluster center)
    csf_label = np.argmin(km.cluster_centers_) + 1
    csf_mask = (labels == csf_label)

    csf_mask = remove_small_objects(csf_mask, min_size=100)
    csf_voxels = data[csf_mask].reshape(-1,1)
    return csf_mask, csf_voxels.mean(), csf_voxels.std()


### TO DO
# Needs to include case high contrast value
def volume_CO_inpainting(image_np, seg_CO, val = None):
    mask = seg_CO.astype(bool)
    if not val:
        _, csf_voxels_mean, csf_voxels_std = CSF_Kmeans_segmentor(image_np)

        fill_vol = np.random.normal(loc=csf_voxels_mean, scale=csf_voxels_std, size=image_np.shape)
        fill_vol = gaussian_filter(fill_vol, sigma=1)
    else: 
        fill_vol = np.ones(image_np.shape)*val

    image_np_out = image_np.copy()
    image_np_out[mask] = fill_vol[mask]

    return image_np_out

### TO DO
# Needs to include reverse/not reverse + There is a small difference xx ?
def pair_generator(pairs_list, batch_size=1, cavity_label=None, fill_val=None, compute_inpainting=True, load_CE_mask=False):
    INPUT_SHAPE = (160, 192, 224, 1)
    while True:
        random.shuffle(pairs_list)
        if not cavity_label:
            for i in range(0, len(pairs_list), batch_size):
                batch = pairs_list[i:i+batch_size]
                moving_batch, fixed_batch, moving_seg_batch, fixed_seg_batch = [], [], [], []
                for item in batch:
                    moving = load_and_preprocess_nifti(item["moving_image"], INPUT_SHAPE[:3])
                    fixed = load_and_preprocess_nifti(item["fixed_image"], INPUT_SHAPE[:3])
                    moving_seg = load_all_tumor_mask(item["moving_seg"], INPUT_SHAPE[:3])
                    fixed_seg = load_all_tumor_mask(item["fixed_seg"], INPUT_SHAPE[:3])
                    moving_batch.append(moving)
                    fixed_batch.append(fixed)
                    moving_seg_batch.append(moving_seg)
                    fixed_seg_batch.append(fixed_seg)
                yield ([np.concatenate(moving_batch, axis=0), 
                        np.concatenate(fixed_batch, axis=0),
                        np.concatenate(moving_seg_batch, axis=0),
                        np.concatenate(fixed_seg_batch, axis=0)], 
                    np.concatenate(fixed_batch, axis=0))
        else:
            for i in range(0, len(pairs_list), batch_size):
                batch = pairs_list[i:i+batch_size]
                moving_batch, fixed_batch, moving_seg_batch, fixed_seg_batch, moving_CO_seg_batch, fixed_CO_seg_batch = [], [], [], [], [], []
                for item in batch:
                    moving = load_and_preprocess_nifti(item["moving_image"], INPUT_SHAPE[:3])
                    fixed = load_and_preprocess_nifti(item["fixed_image"], INPUT_SHAPE[:3])
                    if load_CE_mask:
                        moving_seg = load_cavity_mask(item["moving_seg"], INPUT_SHAPE[:3], cavity_label=3)
                        fixed_seg = load_cavity_mask(item["fixed_seg"], INPUT_SHAPE[:3], cavity_label=3)
                    else:
                        moving_seg = load_all_tumor_mask(item["moving_seg"], INPUT_SHAPE[:3])
                        fixed_seg = load_all_tumor_mask(item["fixed_seg"], INPUT_SHAPE[:3])
                    moving_CO_seg = load_cavity_mask(item["moving_seg"], INPUT_SHAPE[:3], cavity_label=4)
                    fixed_CO_seg = load_cavity_mask(item["fixed_seg"], INPUT_SHAPE[:3], cavity_label=4)
                    if compute_inpainting:
                        fixed = volume_CO_inpainting(fixed, fixed_CO_seg, val=fill_val) # xxx ?
                        moving = volume_CO_inpainting(moving, moving_CO_seg, val=fill_val)

                    moving_batch.append(moving)
                    fixed_batch.append(fixed)
                    moving_seg_batch.append(moving_seg)
                    fixed_seg_batch.append(fixed_seg)
                    moving_CO_seg_batch.append(moving_CO_seg)
                    fixed_CO_seg_batch.append(fixed_CO_seg)

                yield ([np.concatenate(moving_batch, axis=0), 
                        np.concatenate(fixed_batch, axis=0),
                        np.concatenate(moving_seg_batch, axis=0),
                        np.concatenate(fixed_seg_batch, axis=0),
                        np.concatenate(moving_CO_seg_batch, axis=0),
                        np.concatenate(fixed_CO_seg_batch, axis=0)], 
                    np.concatenate(fixed_batch, axis=0))



# Train/Validation Split
def split_pairs(pair_list, val_fraction=0.2, seed=42):
    random.seed(seed)

    patient_groups = defaultdict(list)
    for item in pair_list:
        patient_groups[item["patientName"]].append(item)

    patient_ids = list(patient_groups.keys())
    random.shuffle(patient_ids)

    val_size = int(len(patient_ids) * val_fraction)
    val_ids = set(patient_ids[:val_size])
    train_ids = set(patient_ids[val_size:])

    train_pairs = []
    val_pairs = []

    for pid in patient_ids:
        if pid in train_ids:
            train_pairs.extend(patient_groups[pid])
        else:
            val_pairs.extend(patient_groups[pid])
    return train_pairs, val_pairs




# --- Helper metrics ---

def dice_score(seg1, seg2, voxel_threshold=10):
    seg1 = seg1 > 0  # Binarize
    seg2 = seg2 > 0

    fixed_voxel_count = np.sum(seg1)

    if fixed_voxel_count < voxel_threshold:
        return np.nan  # or return None or 0, depending on how you want to handle it

    intersection = np.sum(seg1 & seg2)
    return 2. * intersection / (np.sum(seg1) + np.sum(seg2) + 1e-5)


def hausdorff_distance(seg1, seg2):
    seg1_points = np.argwhere(seg1 > 0)
    seg2_points = np.argwhere(seg2 > 0)
    if len(seg1_points) == 0 or len(seg2_points) == 0:
        return np.nan
    hd1 = np.max([np.min(np.linalg.norm(p - seg2_points, axis=1)) for p in seg1_points])
    hd2 = np.max([np.min(np.linalg.norm(p - seg1_points, axis=1)) for p in seg2_points])
    return max(hd1, hd2)

def volume_diff(seg1, seg2):
    seg1 = seg1 > 0  # Binarize
    seg2 = seg2 > 0
    volume_diff = np.abs(np.sum(seg1) - np.sum(seg2))
    return volume_diff

def ncc(img1, img2):
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    return np.corrcoef(img1_flat, img2_flat)[0, 1]

def mutual_information(img1, img2, bins=64):
    hgram, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    return mutual_info_score(None, None, contingency=hgram)

def compute_ssim(img1, img2):
    img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1) + 1e-8)
    img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2) + 1e-8)
    mid_slice = img1.shape[2] // 2
    return ssim(img1[:, :, mid_slice], img2[:, :, mid_slice], data_range=1.0)

def jacobian_determinant(displacement_field):
    grad_x = np.gradient(displacement_field[..., 0], axis=(0, 1, 2))
    grad_y = np.gradient(displacement_field[..., 1], axis=(0, 1, 2))
    grad_z = np.gradient(displacement_field[..., 2], axis=(0, 1, 2))

    J = np.zeros(displacement_field.shape[:-1] + (3, 3))
    J[..., 0, 0] = 1 + grad_x[0]
    J[..., 0, 1] = grad_x[1]
    J[..., 0, 2] = grad_x[2]
    J[..., 1, 0] = grad_y[0]
    J[..., 1, 1] = 1 + grad_y[1]
    J[..., 1, 2] = grad_y[2]
    J[..., 2, 0] = grad_z[0]
    J[..., 2, 1] = grad_z[1]
    J[..., 2, 2] = 1 + grad_z[2]

    jac_det = np.linalg.det(J)
    return jac_det

# --- Main evaluation function ---

def evaluate_registration(fixed_img, moved_img,
                          fixed_cavity, moved_cavity,
                          fixed_ce, moved_ce,
                          displacement_field):
    results = {}

    # Image similarity metrics
    results['NCC'] = ncc(fixed_img, moved_img)
    results['MI'] = mutual_information(fixed_img, moved_img)
    results['SSIM'] = compute_ssim(fixed_img, moved_img)

    # Cavity segmentation alignment
    results['Dice_Cavity'] = dice_score(fixed_cavity, moved_cavity)
    results['Volume_diff_cavity'] = volume_diff(fixed_cavity, moved_cavity)

    # CE tumor volume preservation
    vol_fixed_ce = np.sum(fixed_ce > 0)
    vol_moved_ce = np.sum(moved_ce > 0)
    results['CE_Volume_Fixed'] = vol_fixed_ce
    results['CE_Volume_Moved'] = vol_moved_ce

    # Jacobian determinant statistics
    jac = jacobian_determinant(displacement_field)
    results['Jacobian_Mean'] = np.mean(jac)
    results['Jacobian_Min'] = np.min(jac)
    results['Jacobian_Max'] = np.max(jac)
    results['Jacobian_Nonpositive'] = np.sum(jac <= 0)

    # Jacobian stats inside CE region
    if np.any(fixed_ce > 0):
        jac_ce = jac[fixed_ce > 0]
        results['Jacobian_CE_Mean'] = np.mean(jac_ce)
        results['Jacobian_CE_Min'] = np.min(jac_ce)
        results['Jacobian_CE_Max'] = np.max(jac_ce)

    return results
