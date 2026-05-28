from scipy.spatial.distance import directed_hausdorff
import tensorflow as tf
import numpy as np

# Mean Squared Error
def compute_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Normalized Cross Correlation
def compute_ncc(y_true, y_pred):
    mean_t = tf.reduce_mean(y_true)
    mean_p = tf.reduce_mean(y_pred)
    y_true_centered = y_true - mean_t
    y_pred_centered = y_pred - mean_p
    numerator = tf.reduce_sum(y_true_centered * y_pred_centered)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered)) * tf.reduce_sum(tf.square(y_pred_centered))) + 1e-5
    return numerator / denominator

# Mean SSIM over depth slices
def compute_ssim(y_true, y_pred):
    depth = tf.shape(y_true)[1]
    total_ssim = 0.0
    for i in range(depth):
        ssim_val = tf.image.ssim(y_true[:, i, :, :, :], y_pred[:, i, :, :, :], max_val=1.0)
        total_ssim += tf.reduce_mean(ssim_val)
    return total_ssim / tf.cast(depth, tf.float32)

# Dice
def dice_coefficient_np(y_true, y_pred, smooth=1e-5):
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

# Hausdorff 
def hausdorff_np(y_true, y_pred):
    y_true_pts = np.argwhere(y_true > 0)
    y_pred_pts = np.argwhere(y_pred > 0)
    if len(y_true_pts) == 0 or len(y_pred_pts) == 0:
        return np.nan
    d1 = directed_hausdorff(y_true_pts, y_pred_pts)[0]
    d2 = directed_hausdorff(y_pred_pts, y_true_pts)[0]
    return max(d1, d2)
