import torch
import torch.nn as nn
import math
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset
import hydra
import os
import logging
from omegaconf import DictConfig, OmegaConf

from src import model

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
log = logging.getLogger(__name__)

@hydra.main(config_path='configs', config_name='config', version_base = None)
def model_creation(cfg: DictConfig):

    print(f"Current working directory: {os.getcwd()}")
    print(OmegaConf.to_yaml(cfg))
    log.info(f"Current working directory: {os.getcwd()}")
    log.info("--- Configuration ---")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    log.info("---------------------")
    

    vit_model = model.VisionTransformer(in_channels=cfg.model.in_channels,
                                        image_size=cfg.model.image_size,
                                        patch_size=cfg.model.patch_size,
                                        number_of_encoder=cfg.model.number_of_encoder,
                                        embeddings = cfg.model.embedding_dims,
                                        d_ff_scale=cfg.model.d_ff_scale_factor,
                                        heads = cfg.model.heads,
                                        input_dropout_rate=cfg.model.input_dropout_rate,
                                        attention_dropout_rate=cfg.model.attention_dropout_rate,
                                        feed_forward_dropout_rate=cfg.model.feed_forward_dropout_rate,
                                        number_of_classes=cfg.model.number_of_classes)
    test_input = torch.randn((cfg.train.batch_size,
                             cfg.model.in_channels,
                             cfg.model.image_size,
                             cfg.model.image_size))
    print(summary(vit_model,input_data= test_input))


if __name__ == "__main__":
    model_creation()