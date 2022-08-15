import torch
from dataset import MelDataset
import sys
from utils import save_checkpoint, load_checkpoint, save_pickle_file
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
# from torchvision.utils import save_image
from discriminator_ViT import ViT
from generator_CNN import Generator
from torch.utils.tensorboard import SummaryWriter
import os
import torch.autograd as autograd

autograd.set_detect_anomaly(True) # detects why it makes Nan loss

def train_fn(disc_Y, disc_X, gen_X, gen_Y, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    Y_reals = 0
    Y_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminators X and Y

        fake_y = gen_Y(x)
        D_Y_real = disc_Y(y)
        
        D_Y_fake = disc_Y(fake_y.detach())
        Y_reals += D_Y_real.mean().item()
        Y_fakes += D_Y_fake.mean().item()
        D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
        D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
        D_Y_loss = D_Y_real_loss + D_Y_fake_loss

        fake_x = gen_X(y)
        D_X_real = disc_X(x)
        D_X_fake = disc_X(fake_x.detach())
        D_X_real_loss = mse(D_X_real, torch.ones_like(D_X_real))
        D_X_fake_loss = mse(D_X_fake, torch.zeros_like(D_X_fake))
        D_X_loss = D_X_real_loss + D_X_fake_loss

        # put it togethor
        D_loss = (D_Y_loss + D_X_loss)/2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        
        # Train Generators X and Y

        # adversarial loss for both generators
        D_Y_fake = disc_Y(fake_y)
        D_X_fake = disc_X(fake_x)
        loss_G_H = mse(D_Y_fake, torch.ones_like(D_Y_fake))
        loss_G_Z = mse(D_X_fake, torch.ones_like(D_X_fake))

        # cycle loss
        cycle_x = gen_X(fake_y)
        cycle_y = gen_Y(fake_x)
        cycle_x_loss = l1(x, cycle_x)
        cycle_y_loss = l1(y, cycle_y)

        # identity loss (remove these for efficiency if you set lambda_identity=0)
        identity_x = gen_X(x)
        identity_y = gen_Y(y)
        identity_x_loss = l1(x, identity_x)
        identity_y_loss = l1(y, identity_y)

        # total G loss
        G_loss = (
            loss_G_Z
            + loss_G_H
            + cycle_x_loss * config.LAMBDA_CYCLE
            + cycle_y_loss * config.LAMBDA_CYCLE
            + identity_y_loss * config.LAMBDA_IDENTITY
            + identity_x_loss * config.LAMBDA_IDENTITY
        )

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 100 == 0:
            save_pickle_file(variable = fake_x, fileName = os.path.join("saved_mels", f"{idx}_mel_x.pickle"))
            # save_pickle_file(variable = fake_x *0/5 + 0.5, fileName = os.path.join("saved_mels", f"{idx}_mel_x.pickle"))

            save_pickle_file(variable = fake_y, fileName = os.path.join("saved_mels", f"{idx}_mel_y.pickle"))
            # save_pickle_file(variable = fake_y *0/5 + 0.5, fileName = os.path.join("saved_mels", f"{idx}_mel_y.pickle"))

            writer_G_Loss = SummaryWriter(f"/home/yuholee/develop/ViTCycleGAN_Mel/logs")
            writer_D_Loss = SummaryWriter(f"/home/yuholee/develop/ViTCycleGAN_Mel/logs")

            writer_G_Loss.add_scalar("G-Loss", G_loss, idx)
            writer_D_Loss.add_scalar("D-Loss", D_loss, idx)

        loop.set_postfix(D_loss=D_loss.item()/(idx+1), G_loss=G_loss.item()/(idx+1))
    


def main():
    disc_Y = ViT(in_channels=1).to(config.DEVICE)
    # disc_Y = Discriminator(in_channels=3).to(config.DEVICE)
    disc_X = ViT(in_channels=1).to(config.DEVICE)
    # disc_X = Discriminator(in_channels=3).to(config.DEVICE)
    gen_X = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)
    gen_Y = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_Y.parameters()) + list(disc_X.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_X.parameters()) + list(gen_Y.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_Y, gen_Y, opt_gen, config.LEARNING_RATE,
        )

        load_checkpoint(
            config.CHECKPOINT_GEN_X, gen_X, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Y, disc_Y, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_X, disc_X, opt_disc, config.LEARNING_RATE,
        )

    dataset = MelDataset(
        root_y="/home/yuholee/develop/ViTCycleGAN_Mel/data_KR/train_mel/B_M_14_t", root_x="/home/yuholee/develop/ViTCycleGAN_Mel/data_KR/train_mel/A_F_02_t"
    )
    val_dataset = MelDataset(
        root_y="/home/yuholee/develop/ViTCycleGAN_Mel/data_KR/eval_mel/B_M_14_e/", root_x="/home/yuholee/develop/ViTCycleGAN_Mel/data_KR/eval_mel/A_F_02_e/"
    )
    # val_dataset = MelDataset(
    #     root_y="/home/yuholee/develop/ViTCycleGAN_Mel/data_KR/eval_mel/B_M_14_e/", root_x="/home/yuholee/develop/ViTCycleGAN_Mel/data_KR/eval_mel/A_F_02_e/", transform=config.transforms
    # )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()


    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_Y, disc_X, gen_X, gen_Y, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        print(f"epoch: {epoch}")
        if config.SAVE_MODEL:
            save_checkpoint(gen_Y, opt_gen, filename=config.CHECKPOINT_GEN_Y)
            save_checkpoint(gen_X, opt_gen, filename=config.CHECKPOINT_GEN_X)
            save_checkpoint(disc_Y, opt_disc, filename=config.CHECKPOINT_CRITIC_Y)
            save_checkpoint(disc_X, opt_disc, filename=config.CHECKPOINT_CRITIC_X)



if __name__ == "__main__":
    main()