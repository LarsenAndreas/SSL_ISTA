import numpy as np
from helper import font
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **font)

training_percentage = 1 # 1 or 80 for 1% or 80% respectively

assert (training_percentage == 1 or training_percentage == 80)
if training_percentage == 80:
    ISTANet = ['ISTANet_80p-flowers_flowers.npy', 'ISTANet_80p-pets_pets.npy']
    ISTA2Net = ['ISTANet_80p_d2v-flowers_flowers.npy', 'ISTANet_80p_d2v-pets_pets.npy']
    MAE = ['maesr-flowers-good_flowers.npy', 'maesr-pets-good_pets.npy']
    MMAE = ['mmaesr-flowers-good_flowers.npy', 'mmaesr-pets-good_pets.npy']
elif training_percentage == 1:
    ISTANet = ['ISTANet_1p-flowers_flowers.npy', 'ISTANet_1p-pets_pets.npy']
    ISTA2Net = ['ISTANet_1p_d2v-flowers_flowers.npy', 'ISTANet_1p_d2v-pets_pets.npy']
    MAE = ['maesr_1p-flowers_flowers.npy', 'maesr_1p-pets_pets.npy']
    MMAE = ['mmaesr_1p-flowers_flowers.npy', 'mmaesr_1p-pets_pets.npy']

model_names = ('ISTA-Net', 'ista2vec', 'MAE', 'ISTA-MAE')
PSNR_means = {
    'Flowers': [],
    'Pets': []
}
PSNR_stds = {
    'Flowers': [],
    'Pets': []
}
stds = []
for file in ISTANet + ISTA2Net + MAE + MMAE:
    array = np.load('results/' + file)
    if len(array) > 2000:
        array = array[int(len(array)*0.8):]
    if 'flowers' in file:
        PSNR_means['Flowers'].append(array.mean())
        PSNR_stds['Flowers'].append(array.std())
    else:
        PSNR_means['Pets'].append(array.mean())
        PSNR_stds['Pets'].append(array.std())
spacing = 0.3
multiplier = 0
x = np.arange(len(model_names))
fig, ax = plt.subplots(layout='constrained')

for means, stds in zip(PSNR_means.items(), PSNR_stds.items()):
    dataset, average = means
    _, std = stds
    if dataset == 'Flowers':
        color = '#2ca02c'
    else:
        color = '#8c564b'
    for mn, a, s in zip(model_names, average, std):
        print(dataset, mn, f'std {s/a:.3f}')
    offset = spacing * multiplier
    rects = ax.bar(x + offset, average, spacing, label=dataset, align='edge', color=color, yerr=std, capsize=5)
    ax.bar_label(rects, padding=0, label_type='center', color='w', fmt='%.1f dB', rotation=90, fontsize=28, weight='bold')
    multiplier += 1
ax.set_ylabel('Average PSNR [dB]')
ax.set_xticks(x + spacing, model_names)
ax.legend(loc='upper center', ncols=3)
plt.show()
