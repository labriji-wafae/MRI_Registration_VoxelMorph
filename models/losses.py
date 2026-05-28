"""
Custom loss functions and regularizers for VoxelMorph deformable registration.
Includes masked intensity losses, bending energy, and structure-guided cavity losses.
"""

import tensorflow as tf
from voxelmorph.tf.losses import NCC, Grad
from voxelmorph.tf.layers import SpatialTransformer

EPS = 1e-6
grad_smooth_routine = Grad(penalty='l2')
stn_lin = SpatialTransformer(interp_method='linear')


def masked_ncc_loss(fixed, moved, mask=None, win=7, corner=8):
    """
    Compute Normalized Cross Correlation (NCC) loss with spatial masking.
    Automatically removes background intensity sampled from an image corner.
    """
    # Élimination automatique de l'arrière-plan en échantillonnant le coin de l'image
    b = tf.reduce_mean(fixed[:, :corner, :corner, :corner, :], axis=[1, 2, 3, 4], keepdims=True)
    background_mask = tf.cast(tf.abs(fixed - b) > 1e-4, fixed.dtype)
    
    if mask is not None:
        m = tf.cast(background_mask * mask, fixed.dtype)
    else:
        m = background_mask

    fixed_m = fixed * m
    moved_m = moved * m

    ncc = NCC(win=win)
    # La NCC de VoxelMorph renvoie nativement une perte (négative) à minimiser
    return ncc.loss(fixed_m, moved_m)


def masked_mse_loss(fixed, moved, mask=None, corner=8):
    """
    Compute Mean Squared Error (MSE) loss restricted to a specific region or mask.
    """
    b = tf.reduce_mean(fixed[:, :corner, :corner, :corner, :], axis=[1, 2, 3, 4], keepdims=True)
    background_mask = tf.cast(tf.abs(fixed - b) > 1e-4, fixed.dtype)
    
    if mask is not None:
        m = tf.cast(background_mask * mask, fixed.dtype)
    else:
        m = background_mask

    error = tf.square(fixed - moved) * m
    return tf.reduce_sum(error) / (tf.reduce_sum(m) + EPS)


def grad_smooth(flow, penalty='l2'):
    """
    Compute the gradient smoothing loss (membrane energy) on the deformation field.
    """
    return grad_smooth_routine.loss(None, flow)


def bending_energy(flow, spacing=(1.0, 1.0, 1.0)):
    """
    Compute Bending Energy regularizer (second-order derivatives of the displacement field)
    to penalize non-fluid, non-smooth unphysical folds.
    """
    # Calcul des dérivées partielles secondes pour contraindre la rigidité de la déformation
    dy, dx, dz = tf.image.image_gradients(flow)
    
    d2y_dy2, _, _ = tf.image.image_gradients(dy)
    _, d2x_dx2, _ = tf.image.image_gradients(dx)
    _, _, d2z_dz2 = tf.image.image_gradients(dz)
    
    be = tf.square(d2y_dy2 / spacing[0]**2) + \
         tf.square(d2x_dx2 / spacing[1]**2) + \
         tf.square(d2z_dz2 / spacing[2]**2)
         
    return tf.reduce_mean(be)


def soft_dice_loss(y_true, y_pred, smooth=1e-5):
    """
    Compute differentiable Soft Dice loss for continuous warped segmentation masks.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3, 4])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3, 4]) + tf.reduce_sum(y_pred, axis=[1, 2, 3, 4])
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)


def compute_loss_from_cfg(fixed, warped, flow, fixed_seg, moving_seg, fixed_CO_seg, moving_CO_seg, cfg_params):
    """
    Combine and weigh the total composite loss according to the config specifications.
    Combines: Similarity Loss + Regularization Penalty + Cavity Alignment Constraint.
    """
    # 1. Extraction des paramètres de configuration de perte
    w_sim = cfg_params["w_sim"]
    w_reg = cfg_params["w_reg"]
    w_cav = cfg_params["w_cav"]
    
    sim_cfg = cfg_params["sim"]
    reg_cfg = cfg_params["reg"]
    cav_cfg = cfg_params["cav"]
    spacing = cfg_params["spacing"]

    # 2. Choix de la métrique de similarité d'intensité
    sim_name = str(sim_cfg["name"]).lower()
    if sim_name == "ncc":
        sim_loss = masked_ncc_loss(fixed, warped, win=int(sim_cfg.get("win", 7)))
    elif sim_name == "mse":
        sim_loss = masked_mse_loss(fixed, warped)
    else:
        raise ValueError(f"Unknown similarity loss metric: {sim_name}")

    # 3. Choix du régularisateur sur le champ de déformation (Velocity/Displacement Field)
    reg_name = str(reg_cfg["name"]).lower()
    if reg_name == "grad":
        reg_loss = grad_smooth(flow, penalty=str(reg_cfg.get("penalty", "l2")))
    elif reg_name == "bending":
        reg_loss = bending_energy(flow, spacing=spacing)
    else:
        raise ValueError(f"Unknown deformation field regularizer: {reg_name}")

    # 4. Perte anatomique ciblée (ici sur la cavité de résection tumorale)
    # Déformation différentiable du masque de la cavité via la couche SpatialTransformer
    warped_cavity_soft = stn_lin([moving_CO_seg, flow])
    cav_name = str(cav_cfg["name"]).lower()
    
    if cav_name == "dice":
        cav_loss = soft_dice_loss(fixed_CO_seg, warped_cavity_soft)
    else:
        raise ValueError(f"Unknown anatomical segmentation loss: {cav_name}")

    # Combinatoire pondérée finale
    total_loss = (w_sim * sim_loss) + (w_reg * reg_loss) + (w_cav * cav_loss)
    return total_loss, sim_loss, reg_loss, cav_loss
