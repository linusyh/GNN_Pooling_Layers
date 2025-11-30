#!/bin/bash
source .env
export HYDRA_FULL_ERROR=1 
export WANDB_START_METHOD=thread 
if [[ -n $DEBUG ]]; then
    exec="python -Xfrozen_modules=off -m debugpy --listen 7730 --wait-for-client"
else
    exec=python
fi
COMMON_TRAIN_PARAMS="extras.enforce_tags=False scheduler=plateau test=True dataset.datamodule.num_workers=0"