import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import color

def center_crop(img, height, width):
    orig_height, orig_width = img.shape[0], img.shape[1]
    padding_height = (orig_height - height) // 2
    padding_width = (orig_width - width) // 2

    if np.ndim(img) == 3:
        return img[padding_height:padding_height+height, padding_width:padding_width+width, :]
    else:
        return img[padding_height:padding_height+height, padding_width:padding_width+width]


def plot_heatmap(rel_vect, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    fig = plt.imshow(rel_vect, cmap=cm.seismic, vmin=-np.max(np.abs(rel_vect)),
                     vmax=np.max(np.abs(rel_vect)), interpolation="nearest")
    fig.axes.get_xaxis().set_ticks([])
    fig.axes.get_yaxis().set_ticks([])
    plt.title('heatmap')
    plt.savefig(output_path)


def plot_overlayed_heatmap(img, rel_vect, output_dir, filename, alpha = 0.8, gray_factor_bg = 0.3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename)
    if (img > 1).any():
        img /= 255
    if img.shape[2] == 3:
        img = color.rgb2gray(img)
    img = 1 - np.dstack((img, img, img)) * gray_factor_bg
    mask = plt.imshow(rel_vect, cmap=cm.seismic, vmin=-np.max(np.abs(rel_vect)),
                      vmax=np.max(np.abs(rel_vect)), interpolation='nearest')
    mask = mask.to_rgba(rel_vect)[:, :, [0, 1, 2]]

    img_hsv = color.rgb2hsv(img)
    mask_hsv = color.rgb2hsv(mask)
    img_hsv[..., 0] = mask_hsv[..., 0]
    img_hsv[..., 1] = mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)
    fig = plt.imshow(img_masked, cmap=cm.seismic, vmin=-np.max(np.abs(img_masked)),
                     vmax=np.max(np.abs(img_masked)), interpolation='nearest')
    fig.axes.get_xaxis().set_ticks([])
    fig.axes.get_yaxis().set_ticks([])
    plt.title('overlaid heatmap')
    plt.savefig(output_path)
    plt.close()
