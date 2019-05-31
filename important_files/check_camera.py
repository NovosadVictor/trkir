import os, re, sys

import numpy as np
from PIL import Image
from datetime import datetime, st

from send_email import send_mail

X_bias_l, X_bias_r = 40, 40


def cut_boundaries(part, index, boundaries):
    return (part[1] >= boundaries[0][index * 2]) & (part[1] <= boundaries[0][index * 2 + 1]) &\
           (part[0] >= boundaries[1][index * 2]) & (part[0] <= boundaries[1][index * 2 + 1])


def split_parts(left_part, right_part, boundaries):
    left_bound = cut_boundaries(left_part, 0, boundaries)
    right_bound = cut_boundaries(right_part, 1, boundaries)

    return [
        [left_part[0][np.where(left_bound)],
         left_part[1][np.where(left_bound)]],
        [right_part[0][np.where(right_bound)],
         right_part[1][np.where(right_bound)]]
    ]


def check_mail(dirname, subdirname):
    print('checking camera for ', dirname, subdirname)
    imgs = [f for f in os.listdir(os.path.join(dirname, subdirname, 'JPEG')) if '.jpg' in f]

    imgs.sort(key=lambda x: int(x.split('.')[0]))
    imgs = imgs[-20:]

    if len(imgs) < 10:
        return

    same = True
    person = False
    leaved_l = False
    leaved_r = False

    for i in range(1, len(imgs)):
        img_2 = np.array(Image.open(os.path.join(dirname, subdirname, 'JPEG', imgs[i])))
        img_1 = np.array(Image.open(os.path.join(dirname, subdirname, 'JPEG', imgs[i - 1])))
        shape = img_2.shape

        if not np.array_equal(img_1, img_2):
            same = False

        left_tr = list(set(np.sort(img_2.T[: (shape[1] - X_bias_l) // 2], axis=None)))
        right_tr = list(set(np.sort(img_2.T[(shape[1] - X_bias_l) // 2: shape[1] - X_bias_r], axis=None)))

        if left_tr[-1] < 180:
            leaved_l = True
            break
        if right_tr[-1] < 180:
            leaved_r = True
            break

        white_part_left = np.where(img_2 >= left_tr[int(0.6 * len(left_tr))])
        white_part_right = np.where(img_2 >= right_tr[int(0.6 * len(right_tr))])

        boundaries = [[0, (img_2.shape[1] - X_bias_l) // 2, (img_2.shape[1] - X_bias_l) // 2, img_2.shape[1] - X_bias_r],
                      [0, 480, 0, 480]]
        white_parts = split_parts(white_part_left, white_part_right, boundaries)

        if len(white_parts[0][0]) > 1500 or len(white_parts[1][0]) > 1500:
            person = True
            break

    if leaved_l:
        send_mail('Left mouse leaved the cage')
        with open(dirname + '/', 'a') as f:
            f.write()
        return -2
    if leaved_r:
        send_mail('Right mouse leaved the cage')
        return -2
    if person:
        send_mail('Person in front of the camera')
        return -2
    if same:
        send_mail('Camera is not connected\n Same last 20 images in {} directory'.format(subdirname))


if __name__ == '__main__':
    check_mail(sys.argv[1], sys.argv[2])
