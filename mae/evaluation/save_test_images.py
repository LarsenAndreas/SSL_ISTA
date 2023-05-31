import json
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, ToPILImage
from helper import prepare_model

if __name__ == '__main__':
    names = [
        'mmaesr-flowers-good',
        'mmaesr-pets-good',
        'maesr-flowers-good',
        'maesr-pets-good',
        'maesr_1p-flowers',
        'maesr_1p-pets',
        'mmaesr_1p-flowers',
        'mmaesr_1p-pets'
        ]
    data_split = 0.8
    img_size = 128
    data_folder = 'data/flowers-102'
    image_folder = '/jpg'
    with open(data_folder + '/files.json', 'r') as fp:
        files = json.load(fp)
    startId = int(len(files)*data_split)
    files = files[startId:]
    pre_process = Compose([ToTensor(), CenterCrop(size=(img_size, img_size))])
    downsample = Resize(size=(img_size // 2, img_size // 2))
    models = [prepare_model(f'states/{name}/', 'checkpoint.pth', name.split('-')[0].split('_')[0]) for name in names]
    # SAVE IMAGE
    image_id = len(files) - 1
    toPIL = ToPILImage()
    with Image.open(data_folder + image_folder + '/' + files[image_id]) as img_lr:
        img_lr.save('results/images/orig_image.png')
        img_lr = downsample(pre_process(img_lr))
        plt.imsave(
            'results/images/image_lr.png', img_lr.moveaxis(0, -1).numpy()
        )
        img_bil = toPIL(img_lr).resize((img_size, img_size))
    for model, name in zip(models, names):
        pred = model(img_lr.unsqueeze(0)).detach().squeeze().clip(0, 1)
        pred_pil = toPIL(pred)
        if 'flowers' in data_folder.split('/')[1]:
            data = 'flowers'
        else:
            data = 'pets'
        plt.imsave(f'results/images/{name}.png', pred.moveaxis(0, -1).cpu().numpy())
