import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/yuholee/develop/ViTCycleGAN_Mel/data_KR/train_mel"
VAL_DIR = "/home/yuholee/develop/ViTCycleGAN_Mel/data_KR/eval_mel"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_Y = "checkpoints\genh.pth.tar"
CHECKPOINT_GEN_X = "checkpoints\genz.pth.tar"
CHECKPOINT_CRITIC_Y = "checkpoints\critich.pth.tar"
CHECKPOINT_CRITIC_X = "checkpoints\criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)