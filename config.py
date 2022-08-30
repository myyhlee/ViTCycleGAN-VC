import torch
import os

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/yuholee/develop/data_KR_80_librosa/train"
VAL_DIR = "/home/yuholee/develop/data_KR_80_librosa/eval"
BATCH_SIZE = 16
LEARNING_RATE = 2e-6
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_Y = "/home/yuholee/develop/ViTCycleGAN/checkpoints/gen_y.pth.tar"
CHECKPOINT_GEN_X = "/home/yuholee/develop/ViTCycleGAN/checkpoints/gen_x.pth.tar"
CHECKPOINT_DISC_Y = "/home/yuholee/develop/ViTCycleGAN/checkpoints/disc_y.pth.tar"
CHECKPOINT_DISC_X = "/home/yuholee/develop/ViTCycleGAN/checkpoints/disc_x.pth.tar"

# transforms = A.Compose(
#     [
#         A.Resize(width=256, height=256),
#         A.HorizontalFlip(p=0.5),
#         A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
#         ToTensorV2(),
#     ],
#     additional_targets={"image0": "image"},
# )