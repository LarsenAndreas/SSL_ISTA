import sys
import os
import json
from datetime import datetime
from time import perf_counter

import torch
import torch.nn as nn
from pprint import pformat

sys.path.append('scripts/mae')
from util.sr_training import params_to_dict, ImageData
from models_sr_mae import MAESR
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
import parameters_finetuning as pf

class params:
    with open(pf.model_path + 'model_parameters.json', "r") as fp:
        pars = json.load(fp)
    img_size = pars.get('img_size')
    patch_size = pars.get('patch_size')
    in_chans = pars.get('in_chans')
    embed_dim = pars.get('embed_dim')
    depth = pars.get('depth')
    num_heads = pars.get('num_heads')
    decoder_embed_dim = pars.get('decoder_embed_dim')
    decoder_depth = pars.get('decoder_depth')
    decoder_num_heads = pars.get('decoder_num_heads')
    mlp_ratio = pars.get('mlp_ratio')
    norm_pix_loss = False

    def model_params(self):
        params = {}
        for key in self.__dict__:
            if key in self.pars:
                params[key] = self.__dict__[key]
        return params

p = params
print(f"Initilising the dataset...")
with open(os.path.join(*pf.data_path.split('/')[:-1])+"/files.json", "r") as fp:
    files = [f"{pf.data_path}/{img}" for img in json.load(fp)]
files = files[:int(len(files) * pf.data_percent)]
print(f'Training on {len(files)} images') 
dataset_train = ImageData(files, dim=pf.img_size_hr)
train_loader = DataLoader(dataset_train, pf.batch_size)
model = MAESR(pf.img_size_hr, img_size=p.img_size, patch_size=p.patch_size, in_chans=p.in_chans,
                embed_dim=p.embed_dim, depth=p.depth, num_heads=p.num_heads,
                decoder_embed_dim=p.decoder_embed_dim, decoder_depth=p.decoder_depth, decoder_num_heads=p.decoder_num_heads,
                mlp_ratio=p.mlp_ratio, norm_layer=nn.LayerNorm, norm_pix_loss=p.norm_pix_loss)
device = pf.device
if not torch.cuda.is_available() and device == 'cuda':
    print(f'{device} not available, using cpu')
    device = 'cpu'

print("Initilising logging...")
timestamp = datetime.now().strftime("%y-%m-%d_%H;%M;%S")
timestamp = timestamp[3:]
folder = f"output/{timestamp}_mae_sr_finetuning"
os.mkdir(folder)
with open(folder + "/model_parameters.json", "w") as fp:
    json.dump(p.model_params(p), fp, indent=4)
with open(folder + "/hyper_parameters.json", "w") as fp:
    json.dump(params_to_dict(pf), fp, indent=4)

# Load checkpoint
checkpoint = torch.load(pf.model_path + pf.checkpoint)

model_dict = model.state_dict()
pretrained_dict = checkpoint['model']
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'decoder_pred' not in k} # decoder_pred
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()

optimiser = torch.optim.AdamW(model.parameters(), pf.lr, (0.9, 0.95), weight_decay=pf.weight_decay)

model.to(device)
print(f"MAE SR Model Settings:\n{pformat(p.model_params(p))}")
model_total_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
print(f'Model has {model_total_params:.2e} trainable parameters')

print(f"\nStarting fine-tuning")
smallest_loss = 1e6
for epoch in range(1, pf.epochs + 1):
    t = perf_counter()
    print(20 * "#")
    print(f"Starting Epoch: {epoch}/{pf.epochs}")
    print(f"Current Learning Rate: {optimiser.param_groups[0]['lr']:.5}\n")
    model.train()
    loss_mean = 0
    for batch_HR in train_loader:
        optimiser.zero_grad()
        batch_HR = batch_HR.to(device)
        batch_HR = batch_HR.type(torch.cuda.FloatTensor)
        batch_LR = resize(batch_HR, size=[pf.img_size_lr, pf.img_size_lr])
        batch_LR = batch_LR.to(device)
        loss, y = model(batch_LR, batch_HR)
        loss_mean += loss.item()
        loss.backward()
        optimiser.step()
    if epoch < pf.warmup_epochs:
        optimiser.param_groups[0]["lr"] = pf.lr + (pf.warmup_target - pf.lr) / pf.warmup_epochs * (
            epoch + 1
        )
    loss_mean /= len(train_loader)
    if loss_mean < smallest_loss:
        smallest_loss = loss_mean
        to_save = {
            'model': model.state_dict(),
            'optimiser': optimiser.state_dict(),
            'epoch': epoch
        }
        torch.save(to_save, f"{folder}/checkpoint.pth")

    with open(f"{folder}/loss.txt", "a") as fp:
        fp.write(str(loss_mean) + "\n")

    print(f"Average Loss: {loss_mean:.5e} ({perf_counter() - t:.1f}s)")
    print(20 * "#" + "\n")
print("Done!")
