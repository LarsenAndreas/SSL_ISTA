import sys
import os
import json
from datetime import datetime
from time import perf_counter
from math import isnan

import torch
import torch.nn as nn
import numpy as np
from pprint import pformat

sys.path.append('scripts/mae')
import models_mae as mae
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import parameters as p


def params_to_dict():
    param_dict = {}
    for key in p.__annotations__.keys():
        param_dict[key] = p.__dict__[key]
    return param_dict


param_dict = params_to_dict()
print(f"Initilising the dataset...")
transform_train = transforms.Compose([
            transforms.RandomResizedCrop(p.img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset_train = datasets.ImageFolder(os.path.join(p.data_path, 'train'), transform=transform_train)
train_loader = DataLoader(dataset_train, p.batch_size)
model = mae.MaskedAutoencoderViT(img_size=p.img_size, patch_size=p.patch_size, in_chans=p.in_chans,
                 embed_dim=p.embed_dim, depth=p.depth, num_heads=p.num_heads,
                 decoder_embed_dim=p.decoder_embed_dim, decoder_depth=p.decoder_depth, decoder_num_heads=p.decoder_num_heads,
                 mlp_ratio=p.mlp_ratio, norm_layer=nn.LayerNorm, norm_pix_loss=p.norm_pix_loss)
device = p.device
if not torch.cuda.is_available() and device == 'cuda':
    print(f'{device} not available, using cpu')
    device = 'cpu'
model.to(device)
optimiser = torch.optim.AdamW(model.parameters(), p.lr, (0.9, 0.95), weight_decay=p.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, p.cosine_epochs_period)
print(f"MAE Model Settings:\n{pformat(param_dict)}")
model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model has {model_total_params:.2e} trainable parameters')

print("Initilising logging...")
timestamp = datetime.now().strftime("%y-%m-%d_%H;%M;%S")
timestamp = timestamp[3:]
os.mkdir(f"output/{timestamp}_mae")
with open(f"output/{timestamp}_mae/model_parameters.json", "w") as fp:
    json.dump(param_dict, fp, indent=4)

print("\nTraining starting...")
writer = SummaryWriter()
n_iter = 0
smallest_loss = 1e6
for epoch in range(1, p.epochs + 1):
    t = perf_counter()
    print(20 * "#")
    print(f"Starting Epoch: {epoch}/{p.epochs}")
    print(f"Current Learning Rate: {optimiser.param_groups[0]['lr']:.5}\n")
    writer.add_scalar("MISC/LR", optimiser.param_groups[0]["lr"], n_iter)
    model.train()
    loss_mean = 0
    for batch, _ in train_loader:
        optimiser.zero_grad()
        batch = batch.to(device)
        batch = batch.type(torch.cuda.FloatTensor)
        loss, y, mask = model(batch, p.mae_mask_ratio)
        loss_mean += loss.item()
        loss.backward()
        optimiser.step()
    writer.add_scalar("LOSS/PixelMSE", loss, n_iter)
    if epoch < p.warmup_epochs:
        optimiser.param_groups[0]["lr"] = p.lr + (p.warmup_target - p.lr) / p.warmup_epochs * (
            epoch + 1
        )
    #else:
    #    scheduler.step()
    with torch.no_grad():
        loss_mean /= len(train_loader)
        writer.add_scalar("LOSS/EpochAverage", loss_mean, epoch)

    if isnan(loss_mean):
        print(f"Loss is NaN. Terminating training after {epoch/p.epochs+1} epochs!")
        break

    if loss_mean < smallest_loss:
        smallest_loss = loss_mean
        to_save = {
            'model': model.state_dict(),
            'optimiser': optimiser.state_dict(),
     #       'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(to_save, f"output/{timestamp}_mae/checkpoint.pth")

    with open(f"output/{timestamp}_mae/loss.txt", "a") as fp:
        fp.write(str(loss_mean) + "\n")

    print(f"Average Loss: {loss_mean:.5e} ({perf_counter() - t:.1f}s)")
    print(20 * "#" + "\n")
print("Done!")
