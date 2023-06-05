from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def params_to_dict(pf):
    param_dict = {}
    for key in pf.__annotations__.keys():
        param_dict[key] = pf.__dict__[key]
    return param_dict

class ImageData(Dataset):
    def __init__(self, paths, dim: int=128):
        self.paths = paths
        self.pre_process = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(size=(dim, dim))])
    
    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self.pre_process(img)
            return img.float()
    
    def __len__(self):
        return len(self.paths)
