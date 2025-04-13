"""
Main module to load and train the model. This should be the program entry
point.
"""
import copy
import sys
from typing import List, Optional

import graphein
import hydra
import lightning as L
import lovely_tensors as lt
import torch
import torch.nn as nn
import torch_geometric
from graphein.protein.tensor.dataloader import ProteinDataLoader
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from loguru import logger as log
from omegaconf import DictConfig

from proteinworkshop import (
    constants,
    register_custom_omegaconf_resolvers,
    utils,
)
from proteinworkshop.configs import config
from proteinworkshop.models.base import BenchMarkModel

graphein.verbose(False)
lt.monkey_patch()

def test_model(
    cfg: DictConfig, encoder: Optional[nn.Module] = None
):  # sourcery skip: extract-method
    """
    Trains a model from a config.

    If ``encoder`` is provided, it is used instead of the one specified in the
    config.

    1. The datamodule is instantiated from ``cfg.dataset.datamodule``.
    2. The callbacks are instantiated from ``cfg.callbacks``.
    3. The logger is instantiated from ``cfg.logger``.
    4. The trainer is instantiated from ``cfg.trainer``.
    5. (Optional) If the config contains a scheduler, the number of training steps is
         inferred from the datamodule and devices and set in the scheduler.
    6. The model is instantiated from ``cfg.model``.
    7. The datamodule is setup and a dummy forward pass is run to initialise
    lazy layers for accurate parameter counts.
    8. Hyperparameters are logged to wandb if a logger is present.
    9. The model is compiled if ``cfg.compile`` is True.
    10. The model is trained if ``cfg.task_name`` is ``"train"``.
    11. The model is tested if ``cfg.test`` is ``True``.

    :param cfg: DictConfig containing the config for the experiment
    :type cfg: DictConfig
    :param encoder: Optional encoder to use instead of the one specified in
        the config
    :type encoder: Optional[nn.Module]
    """

    log.info(
        f"Instantiating datamodule: <{cfg.dataset.datamodule._target_}..."
    )
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.callbacks.instantiate_callbacks(
        cfg.get("callbacks")
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.loggers.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer...")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    
    log.info("Instantiating model...")
    model: L.LightningModule = BenchMarkModel(cfg)

    if encoder is not None:
        log.info(f"Setting user-defined encoder {encoder}...")
        model.encoder = encoder

    log.info("Initializing lazy layers...")
    with torch.no_grad():
        datamodule.setup(stage="lazy_init")  # type: ignore
        batch = next(iter(datamodule.val_dataloader()))
        log.info(f"Unfeaturized batch: {batch}")
        batch = model.featurise(batch)
        log.info(f"Featurized batch: {batch}")
        log.info(f"Example labels: {model.get_labels(batch)}")
        # Check batch has required attributes
        for attr in model.encoder.required_batch_attributes:  # type: ignore
            if not hasattr(batch, attr):
                raise AttributeError(
                    f"Batch {batch} does not have required attribute: {attr} ({model.encoder.required_batch_attributes})"
                )
        out = model(batch)
        log.info(f"Model output: {out}")
        del batch, out

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.logging_utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch_geometric.compile(model, dynamic=True)

    if cfg.get("test"):
        log.info("Starting testing!")
        if hasattr(datamodule, "test_dataset_names"):
            splits = datamodule.test_dataset_names
            wandb_logger = copy.deepcopy(trainer.logger)

            # load model from a specified checkpoint path
            ckpt_path = "/home/chang/Projects/GNN/ProteinWorkshop/runs/train/runs/schnet_baseline_epoch_50/checkpoints/epoch_049.ckpt"
            model = model.load_from_checkpoint(ckpt_path)

            for i, split in enumerate(splits):
                dataloader = datamodule.test_dataloader(split)
                trainer.logger = False
                log.info(f"Testing on {split} ({i+1} / {len(splits)})...")
                results = trainer.test(
                    model=model, dataloaders=dataloader, ckpt_path="last"
                )[0]
                results = {f"{k}/{split}": v for k, v in results.items()}
                log.info(f"{split}: {results}")
                wandb_logger.log_metrics(results)
        else:
            #trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
            trainer.test(model=model, datamodule=datamodule, ckpt_path="last")


def _main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    utils.extras(cfg)
    cfg = config.validate_config(cfg)
    test_model(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    register_custom_omegaconf_resolvers()
    _main()  # type: ignore