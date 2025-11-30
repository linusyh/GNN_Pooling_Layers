#!/bin/bash
source common.sh

cd "$(dirname "$0")"/..

LR=${LR:-0.0005}
ENCODER=${ENCODER:-gvp}
DECODER=${DECODER:-graph_label_bn_x2}
DATASET=${DATASET:-fold_fold}
WD=${WD:-1e-3}
if [[ -v N_LAYERS ]]; then
  N_LAYER_ARG="encoder.num_layers=$N_LAYERS"
  ENCODER_TEXT="$ENCODER-$N_LAYERS"
else
  N_LAYER_ARG=""
  ENCODER_TEXT=$ENCODER
fi
$exec \
    proteinworkshop/train.py \
    seed=$RANDOM \
    features=ca_everything \
    task=multiclass_graph_classification \
    name="$ENCODER_TEXT/$DATASET/$LR/BN+WD_$WD+x2" \
    dataset=$DATASET \
    encoder=$ENCODER $N_LAYER_ARG \
    decoder=$DECODER \
    optimiser.optimizer.lr=$LR \
    optimiser.optimizer.weight_decay=$WD \
    trainer.max_epochs=70 \
    logger=wandb \
    $COMMON_TRAIN_PARAMS \
    $*
