import json
import numpy as np
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
from tqdm import tqdm
from PIL import Image
from helper import prepare_model

def getPSNR(data_split=0.8, img_size=128, names=["mmaesr-flowers-good_flowers"], Flowers=True, Pets=False):
    pre_process = Compose([ToTensor(), CenterCrop(size=(img_size, img_size))])
    downsample = Resize(size=(img_size // 2, img_size // 2))
    models = [prepare_model(f'states/{name}/', 'checkpoint.pth', name.split('-')[0].split('_')[0]) for name in names]
    datasets = [k for k, v in {"Flowers": Flowers, "Pets": Pets} if v]
    for dataset in datasets:
        if dataset == "Flowers":
            data_folder = 'data/flowers-102'
            image_folder = '/jpg'
        elif dataset == "Pets":
            data_folder = 'data/OxfordPet'
            image_folder = '/images'
        with open(data_folder + '/files.json', 'r') as fp:
            files = json.load(fp)
        startId = int(len(files)*data_split)
        files = files[startId:]

        output = {}
        for name in names:
            output[name] = np.zeros(len(files))
        for i, file in tqdm(enumerate(files)):
            with Image.open(data_folder + image_folder + '/' + file) as img:
                img = pre_process(img)
            img_lr = downsample(img).unsqueeze(0)
            for name, model in zip(names, models):
                pred = model(img_lr).detach().squeeze().clip(0, 1)
                output[name][i] = -10*np.log10(((pred - img)**2).mean())
        for model, name in zip(models, names):
            np.save(f'results/{name}_{dataset.lower()}.npy', output[name])
    return output

if __name__ == '__main__':
    setup = {
        "data_split": 0.8,
        "img_size": 128,
        "names": [
            "mmaesr-flowers-good",
            "mmaesr-pets-good",
            "maesr-flowers-good",
            "maesr-pets-good",
            "maesr_1p-flowers",
            "maesr_1p-pets",
            "mmaesr_1p-flowers",
            "mmaesr_1p-pets"
        ],
        "Flowers": True,
        "Pets": False
    }
    output = getPSNR(**setup)
