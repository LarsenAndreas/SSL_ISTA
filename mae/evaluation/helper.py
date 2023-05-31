import json
import torch
import sys
sys.path.append('.')
from models_sr_istamae import ISTAMAESR as istamaesr
from models_sr_mae import MAESR as maesr

font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 18}

class ISTAMAESR(istamaesr):
    def forward(self, imgs_LR, mask_ratio=0, alpha=1e-2):
        latent, *_, ids_restore = self.forward_encoder(imgs_LR, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, k*p*p*3]
        return self.unpatchify(pred)


class MAESR(maesr):
    def forward(self, imgs_LR, mask_ratio=0):
        latent, _, ids_restore = self.forward_encoder(imgs_LR, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, k*p*p*3]
        return self.unpatchify(pred)


def prepare_model(dir, chkpt, model_name):
    with open(dir + 'model_parameters.json', 'r') as fp:
        model_params = json.load(fp)
    model_params['img_size_hr'] = model_params['img_size'] * 2
    parameters = ["img_size_hr", "img_size", "patch_size", "in_chans", "embed_dim", "depth", "num_heads", "decoder_embed_dim", "decoder_depth", "decoder_num_heads", "mlp_ratio"]
    parameters_full = [*parameters, "embed_depth"]
    if model_name.lower() == 'mmaesr':
        params = {}
        for key in parameters_full:
            params[key] = model_params[key]
        model = ISTAMAESR(**params)
        chkpt = torch.load(dir + chkpt, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['model'])
        model.eval()
    elif model_name.lower() == 'maesr':
        params = {}
        for key in parameters:
            params[key] = model_params[key]
        model = MAESR(**params)
        chkpt = torch.load(dir + chkpt, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['model'])
        model.eval()
    return model
