#!/bin/bash

cd "$(dirname "$0")"

# Base config
CFG="configs/finetune.yaml"

# List of experiments
# Each block = run name + overrides
experiments=(
 "--set logging.run_name=Run_5folds_unfreeze_10_00 model.layers_to_unfreeze=10 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=1"
 "--set logging.run_name=Run_5folds_unfreeze_20_00 model.layers_to_unfreeze=20 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=1"
 "--set logging.run_name=Run_5folds_unfreeze_30_00 model.layers_to_unfreeze=30 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=1"
 
 "--set logging.run_name=Run_5folds_unfreeze_10_01 model.layers_to_unfreeze=10 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=0"
 "--set logging.run_name=Run_5folds_unfreeze_10_02 model.layers_to_unfreeze=10 loss.weights.sim=0 loss.weights.reg=0.01 loss.weights.cavity=1"
 "--set logging.run_name=Run_5folds_unfreeze_10_03 model.layers_to_unfreeze=10 loss.weights.sim=0.5 loss.weights.reg=0.01 loss.weights.cavity=1"
 "--set logging.run_name=Run_5folds_unfreeze_10_04 model.layers_to_unfreeze=10 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=0.5"

 "--set logging.run_name=Run_5folds_unfreeze_20_01 model.layers_to_unfreeze=20 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=0"
 "--set logging.run_name=Run_5folds_unfreeze_20_02 model.layers_to_unfreeze=20 loss.weights.sim=0 loss.weights.reg=0.01 loss.weights.cavity=1"
 "--set logging.run_name=Run_5folds_unfreeze_20_03 model.layers_to_unfreeze=20 loss.weights.sim=0.5 loss.weights.reg=0.01 loss.weights.cavity=1"
 "--set logging.run_name=Run_5folds_unfreeze_20_04 model.layers_to_unfreeze=20 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=0.5"

 "--set logging.run_name=Run_5folds_unfreeze_30_01 model.layers_to_unfreeze=30 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=0"
 "--set logging.run_name=Run_5folds_unfreeze_30_02 model.layers_to_unfreeze=30 loss.weights.sim=0 loss.weights.reg=0.01 loss.weights.cavity=1"
 "--set logging.run_name=Run_5folds_unfreeze_30_03 model.layers_to_unfreeze=30 loss.weights.sim=0.5 loss.weights.reg=0.01 loss.weights.cavity=1"
 "--set logging.run_name=Run_5folds_unfreeze_30_04 model.layers_to_unfreeze=30 loss.weights.sim=1 loss.weights.reg=0.01 loss.weights.cavity=0.5"    
)

# Loop over experiments
for exp in "${experiments[@]}"; do
 echo "================================================"
 echo " Running experiment: $exp"
 echo "================================================"
 
 python -m engine.finetune --cfg "$CFG" $exp
 
 echo " Finished: $exp"
done
 
