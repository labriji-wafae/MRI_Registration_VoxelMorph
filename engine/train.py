from models.losses import compute_loss_from_cfg
import tensorflow as tf
import numpy as np
import mlflow
from omegaconf import OmegaConf


def to_tensor_dict(batch):
    return {k: tf.convert_to_tensor(v) for k, v in batch.items()}


upsample = tf.keras.layers.UpSampling3D(size=(2, 2, 2))
### Train
@tf.function
def train_step(model, batch, opt, loss_params):
    with tf.GradientTape() as tape:
        warped, flow = model([batch["moving"], batch["fixed"]], training=True)
        flow = upsample(flow)
        total, sim, reg, cav = compute_loss_from_cfg(
            batch["fixed"], warped, flow,
            batch["fixed_seg"], batch["moving_seg"],
            batch["fixed_CO"], batch["moving_CO"],
            loss_params
        )
    grads = tape.gradient(total, model.trainable_variables)
    if loss_params["grad_clip"]:
        grads = [tf.clip_by_norm(g, loss_params["grad_clip"]) if g is not None else None for g in grads]
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return total, sim, reg, cav

#train_steps = len(train_pairs) // BATCH_SIZE
def train_one_epoch(model, train_gen, opt, cfg, epoch):

    loss_params = {
        "w_sim": float(cfg.loss.weights.sim),
        "w_reg": float(cfg.loss.weights.reg),
        "w_cav": float(cfg.loss.weights.cavity),
        "sim": OmegaConf.to_container(cfg.loss.similarity, resolve=True),
        "reg": OmegaConf.to_container(cfg.loss.regularizer, resolve=True),
        "cav": OmegaConf.to_container(cfg.loss.cavity, resolve=True),
        "spacing": tuple(cfg.model.get("spacing_mm", [1,1,1])),
        "grad_clip": float(cfg.train.grad_clip) if cfg.train.grad_clip else None,
    }


    totals, sims, regs, cavs = [], [], [], []
    for step in range(cfg.data.steps_per_epoch):        
        inputs, _ = next(train_gen)   # generator yields (inputs, labels)

        # unpack inputs into dict
        batch = {
            "moving": inputs[0],
            "fixed": inputs[1],
            "moving_seg": inputs[2],
            "fixed_seg": inputs[3],
            "moving_CO": inputs[4],
            "fixed_CO": inputs[5],
        }

        batch = to_tensor_dict(batch)

        # print("Debug here ... ")
        total, sim, reg, cav = train_step(model, batch, opt, loss_params)
        totals.append(float(np.mean(total.numpy())))
        sims.append(float(np.mean(sim.numpy())))
        regs.append(float(np.mean(reg.numpy())))
        cavs.append(float(np.mean(cav.numpy())))

    mlflow.log_metrics({
        "train/total": float(np.mean(totals)),
        "train/sim": float(np.mean(sims)),
        "train/reg": float(np.mean(regs)),
        "train/cavity": float(np.mean(cavs))
    }, step=epoch)

    train_total = float(np.mean(totals)) 
    train_sim = float(np.mean(sims))
    train_reg = float(np.mean(regs))
    train_cav = float(np.mean(cavs))
    return train_total, train_sim, train_reg, train_cav
