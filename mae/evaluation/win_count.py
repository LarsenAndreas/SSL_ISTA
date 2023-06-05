import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helper import font
matplotlib.rc('font', **font)

def winPlot(fil1, fil2):
    PSNR_array1 = np.load(fil1)
    PSNR_array2 = np.load(fil2)
    array = PSNR_array1 - PSNR_array2
    if 'flowers' in fil1.split('_')[-1]:
        id_start = 6551
    elif 'pets' in fil1.split('_')[-1]:
        id_start = 5896
    if len(array) > 2000:
        id_start = int(len(array)*0.8)
        array = array[id_start:]
    mask_win = array > 0
    colours = []
    for b in mask_win:
        if b:
            colours.append('g')
        else:
            colours.append('r')
    xaxis = np.arange(id_start + 1, id_start + len(array) + 1)
    upper_stem = np.zeros_like(array)
    upper_stem[mask_win] = array[mask_win]
    lower_stem = np.zeros_like(array)
    lower_stem[np.invert(mask_win)] = array[np.invert(mask_win)]
    fig, ax = plt.subplots(layout='constrained')
    ax.stem(xaxis, upper_stem, linefmt='g', markerfmt='', basefmt='k')
    ax.stem(xaxis, lower_stem, linefmt='r', markerfmt='', basefmt='k')
    ax.hlines([array[mask_win].mean(), array[np.invert(mask_win)].mean()], xmin=xaxis[0], xmax=xaxis[-1], colors=['black', 'black'], ls='dashed')
    ax.set_title(f'{sum(mask_win)/len(mask_win)*100:.1f}% win rate over baseline')
    ax.set_ylabel('PSNR difference [dB]')
    ax.set_xlabel('Image index [$i$]')
    plt.show()

folder = 'results/'
file1 = folder + 'mmaesr-pets-good_flowers' + '.npy'
file2 = folder + 'maesr-pets-good_flowers' + '.npy'
winPlot(file1, file2)
