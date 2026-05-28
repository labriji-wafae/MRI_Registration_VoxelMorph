import tensorflow as tf
from voxelmorph.tf.losses import NCC, Grad
from voxelmorph.tf.layers import SpatialTransformer
from voxelmorph.tf.losses import NCC, Grad

EPS = 1e-6
grad = Grad(penalty='l2')


def masked_ncc_loss(fixed, moved, mask=None, win=7, corner=8):
    """
    Returns NCC *similarity* (higher is better).
    - Auto background removal from ONE fixed-image corner.
    - If `mask` is provided, multiply it in (e.g., tissue-only, cavity exclusion).
    """
    b = tf.reduce_mean(fixed[:, :corner, :corner, :corner, :], axis=[1,2,3,4], keepdims=True)
    M = tf.cast(tf.abs(fixed - b) > 1e-4, fixed.dtype) # This is a simple background mask
    
    if mask is not None:
        m = tf.cast(M * mask, fixed.dtype)
    else:
        m = M

    fixed_m = fixed * m
    moved_m = moved * m
    denom = tf.reduce_mean(m) + EPS  # occupancy

    ncc = NCC(win=win)
    # VXM's NCC returns a loss (negative NCC) ready to minimize
    return ncc.loss(fixed_m, moved_m) #/ denom

def masked_mse_loss(fixed, moved, mask=None, corner=8):
    b = tf.reduce_mean(fixed[:, :corner, :corner, :corner, :], axis=[1,2,3,4], keepdims=True)
    M = tf.cast(tf.abs(fixed - b) > 1e-4, fixed.dtype) # This is a simple background mask
    
    if mask is not None:
        m = tf.cast(M * mask, fixed.dtype)
    else:
        m = M
    num = tf.reduce_sum(tf.square((fixed - moved) * m))
    den = tf.reduce_sum(m) + 1e-8
    return num / den


def soft_dice_loss(fixed_seg, warped_seg, weights=None, smooth=1e-5, min_vox=10.0):
    """
    Soft Dice loss with optional voxel-wise weights.
    Args:
        fixed_seg:   [B, D, H, W, 1]
        warped_seg:  [B, D, H, W, 1]
        weights:     optional [B, D, H, W, 1]
    Returns:
        scalar mean loss over valid samples (where fixed_seg > min_vox)
    """
    fixed_seg  = tf.cast(fixed_seg,  tf.float32)
    warped_seg = tf.cast(warped_seg, tf.float32)
    warped_seg = tf.clip_by_value(warped_seg, 0.0, 1.0)

    if weights is None:
        weights = tf.ones_like(fixed_seg, dtype=tf.float32)
    else:
        weights = tf.cast(weights, tf.float32)

    axes = [1, 2, 3, 4]  # spatial + channel

    intersection = tf.reduce_sum(weights * fixed_seg * warped_seg, axis=axes)
    denominator  = tf.reduce_sum(weights * fixed_seg, axis=axes) + tf.reduce_sum(weights * warped_seg, axis=axes)
    dice = (2. * intersection + smooth) / (denominator + smooth)
    dice_loss = 1. - dice  # per-sample loss

    # Only average over samples with meaningful cavity
    valid = tf.cast(tf.reduce_sum(fixed_seg, axis=axes) > min_vox, tf.float32)  # shape [B]
    num_valid = tf.reduce_sum(valid)

    mean_loss = tf.reduce_sum(dice_loss * valid) / (num_valid + 1e-8)
    return mean_loss


# You assign higher weights to voxels inside this ring when computing the Dice loss, and lower weights elsewhere.
# The result: Dice loss gradients are strongest near the cavity boundary, and have less effect inside the cavity (where changes are expected).
def boundary_ring_weights(cavity_mask_fixed, band_mm=10.0, spacing=(1.0,1.0,1.0),
                          ring_weight=1.0, core_weight=0.2):
    """
    cavity_mask_fixed: [B,D,H,W,1] binary mask of the cavity (in fixed image space)
    Returns a weight map of same shape, where:
      - boundary region gets ring_weight
      - interior (core) gets core_weight
    """
    y = tf.cast(cavity_mask_fixed, tf.float32)

    # Step 1: find boundary using edge detection (max-pool around the cavity)
    edge = tf.cast(tf.nn.max_pool3d(y, [1,3,3,3,1], [1,1,1,1,1], "SAME") - y > 0, tf.float32)

    # Step 2: expand to a ring (band_mm in physical units)
    vox_band = max(1, int(round(band_mm / float(spacing[0]))))  # assume near-isotropic
    k = 2*vox_band + 1
    ring = tf.nn.max_pool3d(edge, [1,k,k,k,1], [1,1,1,1,1], "SAME")

    # Step 3: core = interior of the cavity (still active, just lower weight)
    core = tf.cast(tf.equal(tf.round(y), 1.0), tf.float32)

    # Combine weights
    weights = ring_weight * ring + core_weight * core
    return weights


def grad_smooth(flow, penalty="l2"):
    return Grad(penalty=penalty).loss(None, flow)

# --- weighted grad smoothness ---
def spatial_weighted_grad_smooth(flow, penalty="l2", weights=None): #weights has shape (B, X, Y, Z, 1)
    """
    flow:  (B, X, Y, Z, 3)
    weights: (B, X, Y, Z, 1) or (B, X, Y, Z) or None
    """
    # finite diffs per axis
    dx = flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :]
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dz = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]

    if penalty == "l1":
        gx = tf.reduce_mean(tf.abs(dx))
        gy = tf.reduce_mean(tf.abs(dy))
        gz = tf.reduce_mean(tf.abs(dz))
    else:  # "l2"
        gx = tf.reduce_mean(tf.square(dx))
        gy = tf.reduce_mean(tf.square(dy))
        gz = tf.reduce_mean(tf.square(dz))

    if weights is None:
        # unweighted (classic)
        return (gx + gy + gz) / 3.0

    # align weights to gradient shapes
    def align(w, tgt):
        # crop last line along the corresponding axis to match gradient shape
        sw = tf.shape(w) #shape of weights
        st = tf.shape(tgt) #target tensor grad_dx /dy/dz
        w = w[:, :st[1], :st[2], :st[3], ...]
        return w

    # ensure weights shape is (B, X, Y, Z, 1)
    if tf.rank(weights) == 4:
        weights = weights[..., tf.newaxis]

    wx = align(weights, dx)
    wy = align(weights, dy)
    wz = align(weights, dz)

    if penalty == "l1":
        gx = tf.reduce_sum(wx * tf.abs(dx)) / (tf.reduce_sum(wx) + 1e-8)
        gy = tf.reduce_sum(wy * tf.abs(dy)) / (tf.reduce_sum(wy) + 1e-8)
        gz = tf.reduce_sum(wz * tf.abs(dz)) / (tf.reduce_sum(wz) + 1e-8)
    else:
        gx = tf.reduce_sum(wx * tf.square(dx)) / (tf.reduce_sum(wx) + 1e-8)
        gy = tf.reduce_sum(wy * tf.square(dy)) / (tf.reduce_sum(wy) + 1e-8)
        gz = tf.reduce_sum(wz * tf.square(dz)) / (tf.reduce_sum(wz) + 1e-8)

    return (gx + gy + gz) / 3.0


def bending_energy(flow, spacing=(1.0, 1.0, 1.0)): #avoids local folding and maintains anatomical topology
    """
    Thin-plate bending energy for a 3D vector field.

    Args
    ----
    u:        [B, D, H, W, 3] float32  (DVF or SVF velocity)
    spacing:  (dz, dy, dx) voxel size in mm

    Returns
    -------
    scalar tf.Tensor: mean bending energy over the volume & batch
    """
    dz, dy, dx = [tf.cast(s, flow.dtype) for s in spacing]

    def shift(t, axis, off):
        # Neumann-like padding via SYMMETRIC, then central shift by ±1 voxel
        pads = [[0,0]]*5
        pads[axis] = [1,1]
        tp = tf.pad(t, pads, mode="SYMMETRIC")
        if off == +1:
            sl = [slice(None)]*5
            sl[axis] = slice(2, None)
            return tp[tuple(sl)]
        else:  # -1
            sl = [slice(None)]*5
            sl[axis] = slice(0, -2)
            return tp[tuple(sl)]

    def d2(t, axis, h):
        # second partial: (f(i+1) - 2f(i) + f(i-1)) / h^2
        return (shift(t, axis, +1) - 2.0*t + shift(t, axis, -1)) / (h*h)

    def d2_mix(t, a, b, ha, hb):
        # mixed second: (f(+a,+b) - f(+a,-b) - f(-a,+b) + f(-a,-b)) / (4 ha hb)
        def s2(t, ax, off):
            pads = [[0,0]]*5; pads[ax] = [1,1]
            tp = tf.pad(t, pads, mode="SYMMETRIC")
            sl = [slice(None)]*5
            if off == +1: sl[ax] = slice(2, None)
            else:         sl[ax] = slice(0, -2)
            return tp[tuple(sl)]
        f_pp = s2(s2(t, a, +1), b, +1)
        f_pm = s2(s2(t, a, +1), b, -1)
        f_mp = s2(s2(t, a, -1), b, +1)
        f_mm = s2(s2(t, a, -1), b, -1)
        return (f_pp - f_pm - f_mp + f_mm) / (4.0*ha*hb)

    # second derivatives (axis: 1=z, 2=y, 3=x)
    u_xx = d2(flow, 3, dx)
    u_yy = d2(flow, 2, dy)
    u_zz = d2(flow, 1, dz)

    u_xy = d2_mix(flow, 3, 2, dx, dy)
    u_xz = d2_mix(flow, 3, 1, dx, dz)
    u_yz = d2_mix(flow, 2, 1, dy, dz)

    # sum over components, then mean over space & batch
    term = (
        tf.square(u_xx) + tf.square(u_yy) + tf.square(u_zz)
        + 2.0*(tf.square(u_xy) + tf.square(u_xz) + tf.square(u_yz))
    )
    return tf.reduce_mean(term)


def weighted_bending_energy(flow, spacing=(1.0, 1.0, 1.0), weights=None):
    """
    Weighted thin-plate bending energy for 3D vector fields.

    Args
    ----
    u:        (B, D, H, W, 3) displacement field
    spacing:  (dz, dy, dx)
    weights:  (B, D, H, W, 1) or (B, D, H, W) spatial weights in [0, +inf)
              (higher → stronger regularization)

    Returns
    -------
    scalar tf.Tensor
    """
    dz, dy, dx = [tf.cast(s, flow.dtype) for s in spacing]

    def shift(t, axis, off):
        pads = [[0,0]]*5
        pads[axis] = [1,1]
        tp = tf.pad(t, pads, mode="SYMMETRIC")
        if off == +1:
            sl = [slice(None)]*5; sl[axis] = slice(2, None)
            return tp[tuple(sl)]
        else:
            sl = [slice(None)]*5; sl[axis] = slice(0, -2)
            return tp[tuple(sl)]

    def d2(t, axis, h):
        return (shift(t, axis, +1) - 2.0*t + shift(t, axis, -1)) / (h*h)

    def d2_mix(t, a, b, ha, hb):
        def s2(t, ax, off):
            pads = [[0,0]]*5; pads[ax] = [1,1]
            tp = tf.pad(t, pads, mode="SYMMETRIC")
            sl = [slice(None)]*5
            if off == +1: sl[ax] = slice(2, None)
            else:         sl[ax] = slice(0, -2)
            return tp[tuple(sl)]
        f_pp = s2(s2(t, a, +1), b, +1)
        f_pm = s2(s2(t, a, +1), b, -1)
        f_mp = s2(s2(t, a, -1), b, +1)
        f_mm = s2(s2(t, a, -1), b, -1)
        return (f_pp - f_pm - f_mp + f_mm) / (4.0*ha*hb)

    # Compute all curvature terms
    u_xx = d2(flow, 3, dx)
    u_yy = d2(flow, 2, dy)
    u_zz = d2(flow, 1, dz)
    u_xy = d2_mix(flow, 3, 2, dx, dy)
    u_xz = d2_mix(flow, 3, 1, dx, dz)
    u_yz = d2_mix(flow, 2, 1, dy, dz)

    term = (
        tf.square(u_xx) + tf.square(u_yy) + tf.square(u_zz)
        + 2.0*(tf.square(u_xy) + tf.square(u_xz) + tf.square(u_yz))
    )

    # --- apply weights if provided ---
    if weights is not None:
        if tf.rank(weights) == 4:
            weights = weights[..., tf.newaxis]
        term = term * weights
        # weighted mean (normalize by sum of weights)
        return tf.reduce_sum(term) / (tf.reduce_sum(weights) + 1e-8)

    # uniform case
    return tf.reduce_mean(term)


def binary_dilate(mask, radius_vox):
    """
    mask: (B, X, Y, Z, 1) float/bool
    radius_vox: int dilation radius in voxels (Chebyshev metric)
    """
    k = tf.ones([2*radius_vox+1]*3 + [1,1], dtype=tf.float32)  # (kx, ky, kz, inC=1, outC=1)
    # depthwise 3D conv via conv3d
    # Pad 'SAME' to preserve size; value>0 means at least one neighbor set.
    conv = tf.nn.conv3d(mask, k, strides=[1,1,1,1,1], padding='SAME')
    return tf.cast(conv > 0, tf.float32)

def make_reg_weights(
    tumor_mask, spacing_mm, band_mm=10.0,
    w_core=0.05, w_ring=0.2, w_far=1.0
):
    """
    Returns W_reg in [0,1], low near pathology, high far away.
    cavity_mask, tumor_mask: (B, X, Y, Z, 1) in {0,1}
    spacing_mm: [sx, sy, sz]
    band_mm: ring thickness around cavity in mm
    """
    # union of pathology
    path = tf.clip_by_value(tumor_mask, 0.0, 1.0)

    # convert mm band to voxels (use ceil for coverage)
    sx, sy, sz = [float(s) for s in spacing_mm]
    rx = int(tf.math.ceil(band_mm / sx))
    ry = int(tf.math.ceil(band_mm / sy))
    rz = int(tf.math.ceil(band_mm / sz))
    r = int(max(rx, ry, rz))  # simple isotropic chebyshev ring

    # ring region = dilated(path, r) minus path
    path_dil = binary_dilate(path, r)
    ring = tf.clip_by_value(path_dil - path, 0.0, 1.0)

    # far region = not dilated(path, r)
    far = 1.0 - path_dil

    # compose weights
    W = w_core * path + w_ring * ring + w_far * far
    # ensure shape (B, X, Y, Z, 1)
    if tf.rank(W) == 4:
        W = W[..., tf.newaxis]
    return W


# ---------- the configurable compute_loss ----------

stn_lin = SpatialTransformer(interp_method='linear',  name='stn_lin')   # for loss (diff.)

def compute_loss_from_cfg(
    fixed, warped, flow,
    fixed_seg, moving_seg,
    fixed_CO_seg, moving_CO_seg,
    params,
):
    w_sim = params["w_sim"]
    w_reg = params["w_reg"]
    w_cav = params["w_cav"]
    sim   = params["sim"]
    reg   = params["reg"]
    cav   = params["cav"]
    spacing = params["spacing"]  # voxel spacing used by bending, etc.

    # --- SIMILARITY (drop cavity) ---
    sim_mask = None
    if fixed_seg is not None:
        mask = tf.cast(tf.logical_and(tf.equal(fixed_seg, 0), tf.equal(moving_seg, 0)), tf.float32)
        sim_mask = 1.0 - mask

    name = str(sim["name"]).lower()
    if name == "ncc":
        sim_val  = masked_ncc_loss(fixed, warped, mask=sim_mask, win=int(sim.get("win", 7)))
        sim_loss = tf.reduce_mean(sim_val)
    elif name == "mse":
        sim_loss = -masked_mse_loss(fixed, warped, mask=sim_mask)
    else:
        raise ValueError(f"Unknown similarity: {name}")

    # --- REGULARIZER (spatially weighted) ---
    # Build W_reg using cavity + tumor/edema (CO + GBM/flair) masks
    # Assume you have moving/fixed tumor masks; if not, use zeros of same shape.
    moving_GBM_seg = params.get("moving_GBM_seg", tf.zeros_like(moving_CO_seg))
    fixed_GBM_seg  = params.get("fixed_GBM_seg",  tf.zeros_like(fixed_CO_seg))

    # You can pick either fixed-space or moving-space masks for the weighting; using fixed-space is common.
    # Optionally, warp the moving pathology into fixed to enlarge path region:
    stn_nn = SpatialTransformer(interp_method='nearest', name='stn_nn')  # for masks
    moving_CO_fixed  = stn_nn([moving_CO_seg, flow])
    moving_GBM_fixed = stn_nn([moving_GBM_seg, flow])

    # union pathology in fixed space
    path_CO  = tf.clip_by_value(fixed_CO_seg  + moving_CO_fixed,  0.0, 1.0)
    path_GBM = tf.clip_by_value(fixed_GBM_seg + moving_GBM_fixed, 0.0, 1.0)

    band_mm   = float(reg.get("band_mm", 10.0))  # ring thickness for REGULARIZER
    w_core    = float(reg.get("w_core", 0.05))
    w_ring    = float(reg.get("w_ring", 0.2))
    w_far     = 1.0

    W_reg = make_reg_weights(
        cavity_mask=path_CO,
        tumor_mask=path_GBM,
        spacing_mm=tuple(params.get("spacing_mm", [1,1,1])),
        band_mm=band_mm, w_core=w_core, w_ring=w_ring, w_far=w_far
    )

    name  = str(reg["name"]).lower()
    if name == "grad":
        reg_loss = grad_smooth(flow, penalty=str(reg.get("penalty", "l2")), weights=W_reg)
    elif name == "weighted-grad":
        reg_loss = spatial_weighted_grad_smooth(flow, penalty=str(reg.get("penalty", "l2")), weights=W_reg)
    elif name == "bending":
        reg_loss = bending_energy(flow, spacing=spacing, weights=W_reg)
    elif name == "weighted-bending":
        reg_loss = weighted_bending_energy(flow, spacing=spacing, weights=W_reg)  # implement analogously

    else:
        raise ValueError(f"Unknown regularizer: {reg['name']}")

    # --- CAVITY loss ---
    warped_cavity_soft = stn_lin([moving_CO_seg, flow])
    cname  = str(cav["name"]).lower()
    if cname == "dice":
        cav_loss = soft_dice_loss(fixed_CO_seg, warped_cavity_soft, smooth=1e-5)
    elif cname == "dice_boundary":
        W = boundary_ring_weights(
            fixed_CO_seg,
            band_mm=float(cav.get("band_mm", 10.0)),
            spacing=tuple(params.get("spacing_mm", [1,1,1])),
            ring_weight=float(cav.get("ring_weight", 1.0)),
            core_weight=float(cav.get("core_weight", 0.2)),
        )
        cav_loss = soft_dice_loss(fixed_CO_seg, warped_cavity_soft, weights=W)
    else:
        raise ValueError(f"Unknown cavity loss: {cav['name']}")

    total = w_sim * sim_loss + w_reg * reg_loss + w_cav * cav_loss
    return total, sim_loss, reg_loss, cav_loss
