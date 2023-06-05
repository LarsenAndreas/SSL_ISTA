data_path: str = 'data/OxfordPet/images'
model_path: str = 'states/05-19_10;18;21_mae/'
checkpoint: str = 'checkpoint.pth'

device: str = 'cuda'
batch_size: int = 16
lr: float = 1e-3
warmup_target: float = 1e-2
warmup_epochs: int = 50
weight_decay: float = 0.05
cosine_epochs_period: int = 0
epochs: int = 1000

img_size_hr: int = 128
img_size_lr: int = 64
alpha: float = 0
data_percent: float = 0.01