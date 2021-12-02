import os.path
from os import listdir
import matplotlib.pyplot as plt
from lib import *
import gc
import pickle
from scipy import ndimage
import argparse
import math

def get_common_ker_len_angle(kers):
    max_shape = max([a[0] for a in kers])
    lenghts = [a[0] for a in kers]
    angles =  [a[1] for a in kers]
    
    return (int(np.mean(lenghts)), np.mean(angles))

def rotate_directory(directory_to_rotate, where_to_save_pickle='angles.pickle', where_to_save_imgs='./rotated/'):
    angles = []
    fnames =  listdir(directory_to_rotate)
    gc.enable()
    make_directory(where_to_save_imgs)
    for fname in fnames:
        cur_fname = os.path.join(directory_to_rotate, fname)
        try:
            orig_img = plt.imread(cur_fname)
            img = rgb2gray(orig_img)
            c = Cepstrum(img, batch_size=256, step=0.5)
            a = get_common_ker_len_angle(c.kernels)[1]
            angles.append((a, img.shape))

            rotated_img = ndimage.rotate(orig_img, - a * 180/math.pi)
            # edge = (rotated_img.shape[0] - 600) // 2 + 1
            # plt.imsave(local_save_dir + p, np.clip(rotated_img[edge:edge + 600, edge:edge + 600], 0., 1.))
            plt.imsave(os.path.join(where_to_save_imgs, fname), np.clip(rotated_img, 0., 1.))
            # plt.imsave(os.path.join(where_to_save_imgs, fname), rotated_img)
        except Exception as e:
            print('caught ex', str(e))

    with open(where_to_save_pickle, 'wb') as f:
        pickle.dump(angles, f)

    return


def rerotate_directory(directory_to_rotate, angles_file, where_to_save='./rerotated/'):
    fnames =  listdir(directory_to_rotate)
    with open(angles_file, 'rb') as f:
        angles = pickle.load(f)
    make_directory(where_to_save)
    for idx, fname in enumerate(fnames):
        try:
            a, s = angles[idx]
            img = plt.imread(os.path.join(directory_to_rotate, fname))
            rotated_img = ndimage.rotate(img, a * 180/math.pi)
            edge_x = (rotated_img.shape[0] - s[0]) // 2
            edge_y = (rotated_img.shape[1] - s[1]) // 2
            # print(rotated_img.shape, edge)
            # plt.imsave(os.path.join(local_save_dir,p), np.clip(rotated_img[edge:edge + 600, edge:edge + 600], 0., 1.))
            #plt.imsave(os.path.join(where_to_save, fname), np.clip(rotated_img, 0., 1.))
            plt.imsave(os.path.join(where_to_save, fname), np.clip(rotated_img[edge_x : edge_x + s[0], edge_y : edge_y + s[1]], 0., 1.))
        except Exception as e:
            print('caught ex', str(e))

def add_parser():
    parser = argparse.ArgumentParser(description='Calculate blur angle and rotate image over it.')

    parser.add_argument('action', type=str, nargs='?', help='rotate \'r\'  or rotate back \'rb\' ')
    parser.add_argument('to_r', type=str, nargs='?', help='what directory to rotate')
    parser.add_argument('--pickle', type=str, nargs='?', help='path to angle.pickle', default='angles.pickle')
    parser.add_argument('to_save', type=str, nargs='?', help='what directory to rotate')

    return parser.parse_args()

if __name__ == '__main__':
    args = add_parser()
    if args.action == 'r':
        rotate_directory(args.to_r, args.pickle, args.to_save)
    elif args.action == 'rb':
        rerotate_directory(args.to_r, args.pickle, args.to_save)
