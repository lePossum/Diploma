import numpy as np
from cv2 import imread, imwrite
from scipy import signal
import matplotlib.pyplot as plt
import math
import scipy.stats as st
import sys


def mse(img1, img2):
	return np.sum((img1 - img2)**2) / np.size(img1)


def psnr(img1, img2):
	mse_res = mse(img1, img2)
	if mse_res != 0:
		return 10 * math.log10(1**2 / mse_res)
	else:
		return math.inf


def get_image(image_path, kernel_path):
	img = imread(image_path)
	kernel = imread(kernel_path)
	return img, kernel


def save_image(img, img_finish):
	img = img * 255
	imwrite(img_finish, img)


def convolve(A, z):
	return signal.convolve2d(z, A, boundary='symm', mode='same')

def derivative_main(A, z, u):

	B = convolve(A, z) - u
	res = convolve(np.flip(A), B)

	return res


def derivative_of_stabilizer(z):
	Q = [(1, 0), (0, 1), (1, 1), (1, -1)]
	res = np.zeros(z.shape, dtype=np.float64)

	for x, y in Q:
		a = np.eye(z.shape[0], k=x, dtype=np.float)
		b = np.eye(z.shape[1], k=y, dtype=np.float)
		d = a @ z @ b - z

		sgn = np.sign(d)

		d = np.transpose(a) @ sgn @ np.transpose(b)
		res += (d - sgn) / math.sqrt((x**2 + y**2))

	return res


def derivative_of_stabilizer_B(z):
	Q = [(1, 0), (0, 1), (1, 1), (1, -1)]
	res = np.zeros(z.shape, dtype=np.float64)

	for x, y in Q:
		a = np.eye(z.shape[0], k=x, dtype=np.float)
		b = np.eye(z.shape[1], k=y, dtype=np.float)
		d = a @ z @ b - z

		sgn = np.sign(d)

		d = np.transpose(a) @ sgn @ np.transpose(b) + a @ sgn @ b

		res += (d - 2 * sgn) / math.sqrt((x**2 + y**2))
	return res



def delta_f(z, A, u):
	return 0.0001 * derivative_of_stabilizer(z) + 0.00001 * derivative_of_stabilizer_B(z) + derivative_main(A, z, u)

#77.49
def momentum_method(z, m, A, u):
	z = np.array(z, dtype=np.float64)
	v = np.zeros(z.shape, dtype=np.float64)
	for i in range(1, 100):
		print(i, " iteration")
		g = delta_f(z + m * v, A, u)
		b = (1 * 0.5 ** (i / 100)) / np.sum(g**2)**0.5
		v = m * v - b * g
		z = z + v
	return z




def gkern(nsig):
	kernlen = nsig*6+1

	kern1d = signal.gaussian(kernlen, std=nsig)
	kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
	kernel = kernel_raw/kernel_raw.sum()
	return kernel


# m = 0.85
# blurred, kernel = get_image(sys.argv[1], sys.argv[2])

# if len(blurred.shape) == 3:
#     blurred = blurred[:, :, 0]
#     kernel = kernel[:, :, 0]

# kernel = kernel / np.sum(kernel)
# blurred = blurred / np.max(blurred)

# name = sys.argv[3]
# noise = int(sys.argv[4])

# z_0 = np.zeros(blurred.shape)

# if noise != 0:
#     if noise == 1:
#         blurred = signal.convolve2d(blurred, gkern(1), boundary='symm', mode='same')
#     else:
#         blurred = signal.convolve2d(blurred, gkern(2), boundary='symm', mode='same')


# z = momentum_method(z_0, m, kernel, blurred)

# if noise != 0:
#     if noise == 1:
#         z = momentum_method(z_0, m, gkern(1), z)
#     else:
#         z = momentum_method(z_0, m, gkern(2), z)

# save_image(z, name)

