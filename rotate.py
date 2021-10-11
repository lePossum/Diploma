import os.path
from os import listdir
import matplotlib.pyplot as plt
from lib import *
import gc
import pickle
from scipy import ndimage

def get_common_ker_len_angle(kers):
    max_shape = max([a[0] for a in kers])
    lenghts = [a[0] for a in kers]
    angles =  [a[1] for a in kers]
    
    return (int(np.mean(lenghts)), np.mean(angles))

def rotate_directory(directory_to_rotate, where_to_save_imgs='./rotated/', where_to_save_pickle='angles.pickle'):
    angles = []
    fnames =  listdir(directory_to_rotate)
    gc.enable()
    make_directory(where_to_save_imgs)
    for fname in fnames:
        cur_fname = os.path.join(directory_to_rotate, fname)

        img = rgb2gray(plt.imread(cur_fname))
        c = Cepstrum(img, batch_size=256, step=0.5)
        a = get_common_ker_len_angle(c.kernels)[1]
        angles.append(a)

        rotated_img = ndimage.rotate(img, a)
        # edge = (rotated_img.shape[0] - 600) // 2 + 1
        # plt.imsave(local_save_dir + p, np.clip(rotated_img[edge:edge + 600, edge:edge + 600], 0., 1.))
        plt.imsave(os.path.join(where_to_save_imgs, fname), np.clip(rotated_img, 0., 1.))

    with open(where_to_save_pickle, 'wb') as f:
        pickle.dump(angles, f)

    return



def rerotate_directory(directory_to_rotate, angles, where_to_save='./rerotated/'):
    fnames =  listdir(directory_to_rotate)
    make_directory(where_to_save)
    for idx, fname in fnames:
        a = angles[idx]
        img = plt.imread(os.path.join(directory_to_rotate, fname))
        rotated_img = ndimage.rotate(img, -a)
        # edge = (rotated_img.shape[0] - 600) // 2 + 1
        # print(rotated_img.shape, edge)
        # plt.imsave(os.path.join(local_save_dir,p), np.clip(rotated_img[edge:edge + 600, edge:edge + 600], 0., 1.))
        plt.imsave(os.path.join(where_to_save, fname), np.clip(rotated_img, 0., 1.))

