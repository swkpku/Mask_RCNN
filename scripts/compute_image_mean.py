import argparse
import numpy as np
import os

from skimage import io

mean_0 = .0
mean_1 = .0
mean_2 = .0

N = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help="Directory of images to read.")
    args = parser.parse_args()

    for subdir, dirs, files in os.walk(args.dir):
        for fName in files:
            img = io.imread(os.path.join(subdir, fName))
            img_mean_0 = np.mean(img[:, :, 0])
            img_mean_1 = np.mean(img[:, :, 1])
            img_mean_2 = np.mean(img[:, :, 2])

            mean_0 = mean_0 + img_mean_0
            mean_1 = mean_1 + img_mean_1
            mean_2 = mean_2 + img_mean_2

            N = N + 1

    mean_0 /= N
    mean_1 /= N
    mean_2 /= N

    print(mean_0)
    print(mean_1)
    print(mean_2)

    print(N)

