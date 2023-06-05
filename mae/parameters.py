data_path: str = '../ILSVRC/Data/CLS-LOC/'

device: str = 'cuda'
epochs: int =  100
batch_size: int = 512
mask_ratio: float = 0.75
lr: float = 1e-4
weight_decay: float = 0.05
warmup_epochs: int = 10
warmup_target: float = 2e-3
cosine_epochs_period: int = 15

img_size: int = 64
patch_size: int = 8
in_chans: int = 3
embed_dim: int = 256
embed_depth: int = 5
depth: int = 6
num_heads: int = 8
decoder_embed_dim: int = 128
decoder_depth: int = 2
decoder_num_heads: int = 8
mlp_ratio: float = 4
norm_pix_loss = False
mae_mask_ratio: float = 0.75
mmae_mask_ratio: float = 0.75
alpha: float = 1e-4
