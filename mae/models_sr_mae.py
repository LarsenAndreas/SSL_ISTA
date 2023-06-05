import models_mae as mae
import torch
from numpy import sqrt

class MAESR(mae.MaskedAutoencoderViT):
    def __init__(self, img_size_hr, **kwargs):
        super(MAESR, self).__init__(**kwargs)
        assert img_size_hr % (kwargs['img_size'] // kwargs['patch_size']) == 0
        patch_size_hr = img_size_hr // (kwargs['img_size'] // kwargs['patch_size'])
        self.decoder_pred = torch.nn.Linear(kwargs['decoder_embed_dim'], patch_size_hr**2 * 3, bias=True) # decoder to patch

    def patchify(self, imgs):
        """
        imgs: (N, c, H, W)
        x: (N, L, patch_size**2 *c)
        """
        c = self.in_chans
        if imgs.shape[2] == imgs.shape[3] and imgs.shape[2] != self.patch_embed.img_size[0]:
            p = int(imgs.shape[2] // sqrt(self.patch_embed.num_patches))
        else:
            p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *c)
        imgs: (N, c, H, W)
        """
        c = self.in_chans
        p = int(sqrt(x.shape[-1] // 3))
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
        
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = loss.sum()

        return loss
    
    def forward(self, imgs_LR, imgs_HR, mask_ratio=0):
        latent, _, ids_restore = self.forward_encoder(imgs_LR, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, k*p*p*3]
        loss = self.forward_loss(imgs_HR, pred)
        return pred, loss