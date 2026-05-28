"""
Main script for VoxelMorph fine-tuning.
Handles 5-fold cross-validation, hyperparameter management via OmegaConf,
and experiment tracking with MLflow.
"""

import os
import argparse
import logging
import warnings
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import mlflow

# Imports VoxelMorph
from voxelmorph.tf.layers import SpatialTransformer
from voxelmorph.tf.networks import VxmDense

from data.loader import pair_generator, split_pairs
from engine.train import train_one_epoch
from engine.validation import validate_one_epoch, run_validation_metrics
from engine.plots import plot_losses
from omegaconf import OmegaConf

# Ignorer les messages de dépréciation internes à VoxelMorph
warnings.filterwarnings("ignore", message=".*int_downsize is deprecated.*")
warnings.filterwarnings("ignore", message=".*unet_half_res is deprecated.*")


def load_cfg():
    """
    Load and parse configuration using OmegaConf with CLI overrides support.
    Example: python finetune.py --set train.lr=1e-4 train.epochs=50
    """
    p = argparse.ArgumentParser(description="VoxelMorph Fine-tuning Pipeline")
    p.add_argument("--cfg", type=str, default="configs/finetune.yaml",
                   help="Path to the yaml configuration file.")
    p.add_argument("--set", nargs="*", default=[],
                   help="Allow dotlist overrides (e.g., --set train.lr=3e-4)")
    args = p.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.set:
        overrides = OmegaConf.from_dotlist(args.set)
        cfg = OmegaConf.merge(cfg, overrides)
    
    return cfg


def make_folds(all_pairs, k=5, seed=42):
    """Split dataset pairs into K robust training/validation folds."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return list(kf.split(all_pairs))


def main():
    # 1. Chargement de la configuration
    cfg = load_cfg()
    
    # Configuration du logging
    out_dir = Path(cfg.paths.get("out_dir", "experiments"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "pipeline.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting VoxelMorph fine-tuning pipeline...")

    # 2. Préparation des paires de données (Longitudinal M0/M5)
    # On suppose que le chemin du JSON est configuré dans le YAML
    json_path = cfg.data.get("pairs_json", "MsrGB_pairs_with_seg_T1c.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset JSON file not found at: {json_path}")
        
    with open(json_path, "r") as f:
        all_pairs = json.load(f)
    
    logging.info(f"Loaded {len(all_pairs)} image pairs from dataset JSON.")
    
    # 3. Initialisation du tracking MLflow
    mlflow.set_tracking_uri(cfg.logging.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(cfg.logging.get("experiment_name", "VoxelMorph_Brain_Registration"))

    folds = make_folds(all_pairs, k=cfg.train.get("k_folds", 5), seed=cfg.train.get("seed", 42))

    # 4. Boucle principale sur les Folds de Validation Croisée
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        logging.info(f"\n--- Launching Training for Fold {fold_idx + 1}/{len(folds)} ---")
        
        train_pairs = [all_pairs[i] for i in train_indices]
        val_pairs = [all_pairs[i] for i in val_indices]
        
        # Configuration des générateurs de données TensorFlow custom
        train_gen = pair_generator(train_pairs, batch_size=cfg.train.batch_size, target_shape=tuple(cfg.model.target_shape))
        val_gen = pair_generator(val_pairs, batch_size=cfg.train.batch_size, target_shape=tuple(cfg.model.target_shape))

        run_name = f"{cfg.logging.run_name}_fold{fold_idx + 1}"
        
        with mlflow.start_run(run_name=run_name):
            # Log des hyperparamètres clés dans MLflow
            mlflow.log_params({
                "fold": fold_idx + 1,
                "lr": cfg.train.lr,
                "epochs": cfg.train.epochs,
                "batch_size": cfg.train.batch_size,
                "similarity_loss": cfg.loss.similarity.name,
                "regularizer": cfg.loss.regularizer.name
            })

            # 5. Instanciation du réseau VoxelMorph 3D Dense
            # On charge l'architecture de base (ex: pré-entraînée sur T1)
            base_model_path = cfg.paths.get("base_model", "vxm_dense_brain_T1_3D_mse.h5")
            if not os.path.exists(base_model_path):
                logging.warning(f"Base model checkpoint not found at {base_model_path}. Training from scratch.")
                # Définir ici l'instanciation de base si pas de checkpoint
                model = VxmDense(inshape=tuple(cfg.model.target_shape), nb_unet_features=cfg.model.unet_features)
            else:
                model = keras.models.load_model(
                    base_model_path, 
                    compile=False,
                    custom_objects={"VxmDense": VxmDense, "SpatialTransformer": SpatialTransformer}
                )
            
            # Définition de l'optimiseur
            opt = keras.optimizers.Adam(learning_rate=cfg.train.lr)
            
            history = {
                "train_total": [], "val_total": [],
                "train_sim": [], "val_sim": [],
                "train_reg": [], "val_reg": [],
                "train_cav": [], "val_cav": []
            }
            
            best_val_loss = np.inf
            patience = cfg.train.get("patience", 10)
            wait = 0

            # 6. Boucle d'entraînement par époque
            for epoch in range(cfg.train.epochs):
                logging.info(f"Epoch {epoch + 1}/{cfg.train.epochs}")
                
                # Entraînement sur une époque complète
                train_metrics = train_one_epoch(model, train_gen, opt, cfg, epoch)
                # Validation sur une époque complète
                val_metrics = validate_one_epoch(model, val_gen, cfg, epoch)
                
                # Stockage de l'historique pour affichage des courbes finales
                for k in train_metrics.keys():
                    history[f"train_{k}"].append(train_metrics[k])
                for k in val_metrics.keys():
                    history[f"val_{k}"].append(val_metrics[k])
                
                val_total = val_metrics["total"]
                
                # Stratégie de Early Stopping manuelle propre
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    wait = 0
                    fold_model_dir = out_dir / run_name
                    fold_model_dir.mkdir(parents=True, exist_ok=True)
                    model.save(fold_model_dir / "best_model.h5")
                    logging.info(f" -> Validation loss improved. Best model saved for fold {fold_idx + 1}.")
                else:
                    wait += 1
                    if wait >= patience:
                        logging.info(f"Early stopping triggered at epoch {epoch + 1}. No improvement for {patience} epochs.")
                        break
            
            # 7. Évaluation finale du Fold et sauvegarde des plots
            fold_curve_path = out_dir / run_name / "loss_curves.png"
            plot_losses(history, outpath=str(fold_curve_path))
            mlflow.log_artifact(str(fold_curve_path))
            
            # Calcul des métriques de forme géométrique finales (Dice, Hausdorff, Jacobien)
            best_model = keras.models.load_model(
                out_dir / run_name / "best_model.h5", 
                compile=False,
                custom_objects={"VxmDense": VxmDense, "SpatialTransformer": SpatialTransformer}
            )
            final_geom_metrics = run_validation_metrics(best_model, val_gen, cfg.data.val_steps, cfg)
            mlflow.log_metrics(final_geom_metrics)
            
            logging.info(f"Fold {fold_idx + 1} processing completed. Metrics tracked in MLflow.")


if __name__ == "__main__":
    main()
