import numpy as np
import matplotlib.pyplot as plt

import os
import math

def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale
       Reference: opencv.org
       Formula: grayscale = 0.299*red + 0.587*green + 0.114*blue
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def display_images(imgs,
                   filename,
                   title='',
                   imgs_dir=None,
                   show=False):
    """Display images in an nxn grid

    Arguments:
    imgs (tensor): array of images
    filename (string): filename to save the displayed image
    title (string): title on the displayed image
    imgs_dir (string): directory where to save the files
    show (bool): whether to display the image or not 
          (False during training, True during testing)

    """

    rows = imgs.shape[1]
    cols = imgs.shape[2]
    channels = imgs.shape[3]
    side = int(math.sqrt(imgs.shape[0]))
    assert int(side * side) == imgs.shape[0]

    # create saved_images folder
    if imgs_dir is None:
        imgs_dir = 'saved_images'
    save_dir = os.path.join(os.getcwd(), imgs_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(imgs_dir, filename)
    # rows, cols, channels = img_shape
    if channels==1:
        imgs = imgs.reshape((side, side, rows, cols))
    else:
        imgs = imgs.reshape((side, side, rows, cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title(title)
    if channels==1:
        plt.imshow(imgs, interpolation='none', cmap='gray')
    else:
        plt.imshow(imgs, interpolation='none')
    plt.savefig(filename)
    if show:
        plt.show()
    
    plt.close('all')



#
