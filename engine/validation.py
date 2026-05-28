
from models.metrics import compute_mse, compute_ncc, compute_ssim, dice_coefficient_np, hausdorff_np
from voxelmorph.tf.layers import SpatialTransformer

from models.losses import compute_loss_from_cfg

import tensorflow as tf
import numpy as np
import mlflow



### Validation
def validate_one_epoch(model, val_gen, cfg, epoch):
    totals, sims, regs, cavs = [], [], [], []
    loss_params = {
        "w_sim": float(cfg.loss.weights.sim),
        "w_reg": float(cfg.loss.weights.reg),
        "w_cav": float(cfg.loss.weights.cavity),
        "sim": cfg.loss.similarity,
        "reg": cfg.loss.regularizer,
        "cav": cfg.loss.cavity,
        "spacing": tuple(cfg.model.get("spacing_mm", [1,1,1])),
        "grad_clip": float(cfg.train.grad_clip) if cfg.train.grad_clip else None,
    }

    for step in range(cfg.data.val_steps):
        inputs, _ = next(val_gen)   # generator yields (inputs, labels)

        # unpack inputs into dict
        batch = {
            "moving": inputs[0],
            "fixed": inputs[1],
            "moving_seg": inputs[2],
            "fixed_seg": inputs[3],
            "moving_CO": inputs[4],
            "fixed_CO": inputs[5],
        }
        warped, flow = model([batch["moving"], batch["fixed"]], training=False)
        flow = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(flow)
        total, sim, reg, cav = compute_loss_from_cfg(
            batch["fixed"], warped, flow,
            batch["fixed_seg"], batch["moving_seg"],
            batch["fixed_CO"], batch["moving_CO"],
            loss_params
        )
        totals.append(float(np.mean(total.numpy())))
        sims.append(float(np.mean(sim.numpy())))
        regs.append(float(np.mean(reg.numpy())))
        cavs.append(float(np.mean(cav.numpy())))
        
    mlflow.log_metrics({
        "val/total": float(np.mean(totals)),
        "val/sim": float(np.mean(sims)),
        "val/reg": float(np.mean(regs)),
        "val/cavity": float(np.mean(cavs))
    }, step=epoch)
    val_total = float(np.mean(totals)) #Return val loss for early stopping
    val_sim = float(np.mean(sims))
    val_reg = float(np.mean(regs))
    val_cav = float(np.mean(cavs))

    return val_total, val_sim, val_reg, val_cav
#        totals.append(float(np.mean(total.numpy())))

def run_validation_metrics(model, val_gen, steps, cfg):
    stn = SpatialTransformer(interp_method='nearest', name='val_stn')

    total_loss = total_sim = total_reg = total_cav = 0.0
    total_mse = total_ncc = total_ssim = 0.0
    total_dice = total_hausdorff = 0.0
    valid_cases = 0

    loss_params = {
        "w_sim": float(cfg.loss.weights.sim),
        "w_reg": float(cfg.loss.weights.reg),
        "w_cav": float(cfg.loss.weights.cavity),
        "sim": cfg.loss.similarity,
        "reg": cfg.loss.regularizer,
        "cav": cfg.loss.cavity,
        "spacing": tuple(cfg.model.get("spacing_mm", [1,1,1])),
        "grad_clip": float(cfg.train.grad_clip) if cfg.train.grad_clip else None,
    }

    for _ in range(steps):
        inputs, _ = next(val_gen)   # generator yields (inputs, labels)

        # unpack inputs into dict
        batch = {
            "moving": inputs[0],
            "fixed": inputs[1],
            "moving_seg": inputs[2],
            "fixed_seg": inputs[3],
            "moving_CO": inputs[4],
            "fixed_CO": inputs[5],
        }
        warped, flow = model([batch["moving"], batch["fixed"]], training=False)
        flow = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(flow)
        warped_CO_seg = stn([batch["moving_CO"], flow])

        # Loss terms (use your cfg-based loss, not hard-coded lambdas)
        loss, sim, reg, cav = compute_loss_from_cfg(
            batch["fixed"], warped, flow,
            batch["fixed_seg"], batch["moving_seg"],
            batch["fixed_CO"], batch["moving_CO"],
            loss_params
        )
        #compute_mse, compute_ncc, compute_ssim, dice_coefficient_np, hausdorff_np
        # Image similarity metrics
        mse_val = compute_mse(batch["fixed"], warped)
        ncc_val = compute_ncc(batch["fixed"], warped)
        ssim_val = compute_ssim(batch["fixed"], warped)

        total_loss += float(np.mean(loss.numpy()))
        total_sim  += float(np.mean(sim.numpy()))
        total_reg  += float(np.mean(reg.numpy()))
        total_cav  += float(np.mean(cav.numpy()))
        total_mse  += float(np.mean(mse_val.numpy()))
        total_ncc  += float(np.mean(ncc_val.numpy()))
        total_ssim += float(np.mean(ssim_val.numpy()))

        # Mask-based metrics
        if hasattr(batch["fixed_CO"], "numpy"):
            fixed_np  = (batch["fixed_CO"].numpy()  > 0.5)
        else:
            fixed_np  = (batch["fixed_CO"]  > 0.5)

        if hasattr(warped_CO_seg, "numpy"):
            warped_np = (warped_CO_seg.numpy() > 0.5)
        else:
            warped_np = (warped_CO_seg > 0.5)

        for i in range(warped_np.shape[0]):
            dice = dice_coefficient_np(fixed_np[i, ..., 0], warped_np[i, ..., 0])
            haus = hausdorff_np(fixed_np[i, ..., 0], warped_np[i, ..., 0])
            if not np.isnan(haus):
                total_dice += dice
                total_hausdorff += haus
                valid_cases += 1

    return {
        "val_loss": total_loss / steps,
        "val_sim": total_sim / steps,
        "val_reg": total_reg / steps,
        "val_cav": total_cav / steps,
        "mse": total_mse / steps,
        "ncc": total_ncc / steps,
        "ssim": total_ssim / steps,
        "dice": total_dice / valid_cases if valid_cases > 0 else 0.0,
        "hausdorff": total_hausdorff / valid_cases if valid_cases > 0 else 0.0
    }