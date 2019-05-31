from __future__ import print_function
from __future__ import division
import cv2 as cv
import os, sys
import csv
import numpy as np
import time
from math import atan2, cos, sin, sqrt, pi


import os
os.putenv('DISPLAY', ':0.0')

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    ## [visualization1]


def getOrientation(img, pts):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))

    return cntr, eigenvectors


def get_head(bw, contour, cntr, ev, max_pix_arg):
    p1 = [cntr[0], cntr[1]]
    p2 = p1[:]

    ev = ev[0]
    if ev[0] * ev[1] > 0:
        ev = [abs(ev[0]), abs(ev[1])]
    else:
        if ev[0] > 0:
            ev = [-ev[0], -ev[1]]

    if ev[1] / ev[0] > 50:
        ev = [0, 1]
    elif ev[0] / ev[1] > 50:
        ev = [1, 0]

    # drawAxis(bw, cntr, p1, (0, 0, 255), 2)
    # cv.imshow("image", bw)
    # cv.waitKey();

    try:
        while bw[int(p1[1]), int(p1[0])] != 0:
            p1 = [p1[0] + ev[0], p1[1] + ev[1]]
        while bw[int(p2[1]), int(p2[0])] != 0:
            p2 = [p2[0] - ev[0], p2[1] - ev[1]]

    except:
 #       print('p1, p2')
        return p1, p2
    # drawAxis(bw, cntr, p1, (0, 0, 255), 2)
    # cv.imshow("image", bw)
    # cv.waitKey();

    return p1, p2


def check_contours(m, dirname, img_name, src_m, prev_center=None, prev_head=None, prev_tail=None):
    # Convert image to grayscale
    gray = cv.cvtColor(src_m, cv.COLOR_BGR2GRAY)

    # Convert image to binary
    blur = cv.GaussianBlur(gray, (15, 15), 0)
    _, bw = cv.threshold(blur, np.max([0.6 * np.max(src_m), 100]), 255, cv.THRESH_BINARY)  # + cv.THRESH_OTSU)

    # [pre-process]

    # [contours]
    # Find all the contours in the thresholded image
    _, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    info = []
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(c)
        # Ignore contours that are too small or too large
    #    print(i, img_name, area)
        if area < 2 * 1e2 or 3 * 1e3 < area:
            continue

        #    Find the orientation of each shape
        cntr, ev = getOrientation(src_m, c)
        info.append([area, cntr, ev, c])

    result = []
    for values in info:
        area, cntr, ev, c = values

        zero_img = np.zeros_like(gray)
        cv.drawContours(zero_img, [c], -1, color=255, thickness=-1)

        indexes = np.where(zero_img == 255)
        ppc = zero_img
        ppc[indexes] = gray[indexes]

        # cv.imshow("image", ppc)
        # cv.waitKey();

        max_pix_arg = np.unravel_index(ppc.argmax(), ppc.shape)
        head, tail = get_head(bw, c, cntr, ev, max_pix_arg)

        head, tail = tail, head
        # if np.linalg.norm(np.array(head) - np.array(max_pix_arg)) > np.linalg.norm(np.array(tail) - np.array(max_pix_arg)):
        #    head, tail = tail, head

        body = cntr
        temp_pix = gray[max_pix_arg]

        cv.circle(src_m, (int(head[0]), int(head[1])), 5, (255, 0, 255), 3)
        cv.circle(src_m, (int(body[0]), int(body[1])), 5, (255, 0, 0), 3)
        cv.circle(src_m, (int(tail[0]), int(tail[1])), 5, (0, 255, 0), 3)

        result.append([area, head, body, tail, temp_pix])

    if not os.path.isdir(os.path.join('data', dirname, 'check')):
        os.mkdir(os.path.join('data', dirname, 'check'))

    cv.imwrite(os.path.join('data', dirname, 'check', str(m) + '_ ' + img_name), src_m)
    return result


def tracking(dirname):
    imgs = os.listdir(os.path.join('data', dirname, 'JPEG'))
    imgs.sort(key=lambda x: int(x.split('.')[0]))

    l_t = [[10, 20], [320, 225]]
    r_t = [[325, 25], [640, 220]]
    l_b = [[15, 225], [320, 395]]
    r_b = [[330, 225], [640, 395]]

    global_res = [['area_1_1', 'head_1_1', 'body_1_1', 'tail_1_1', 'temp_1_1',
                   'area_1_2', 'head_1_2', 'body_1_2', 'tail_1_2', 'temp_1_2',
                   'area_2_1', 'head_2_1', 'body_2_1', 'tail_2_1', 'temp_2_1',
                   'area_2_2', 'head_2_2', 'body_2_2', 'tail_2_2', 'temp_2_2',
                   'area_3_1', 'head_3_1', 'body_3_1', 'tail_3_1', 'temp_3_1',
                   'area_3_2', 'head_3_2', 'body_3_2', 'tail_3_2', 'temp_3_2',
                   'area_4_1', 'head_4_1', 'body_4_1', 'tail_4_1', 'temp_4_1',
                   'area_4_2', 'head_4_2', 'body_4_2', 'tail_4_2', 'temp_4_2',
                   'img_name']]

    for i, img_name in enumerate(imgs):
        start = time.clock()

        src = cv.imread(os.path.join('data', dirname, 'JPEG', img_name))

        srcs = [src[l_t[0][1]: l_t[1][1], l_t[0][0]: l_t[1][0]],
                src[r_t[0][1]: r_t[1][1], r_t[0][0]: r_t[1][0]],
                src[l_b[0][1]: l_b[1][1], l_b[0][0]: l_b[1][0]],
                src[r_b[0][1]: r_b[1][1], r_b[0][0]: r_b[1][0]],
                ]

        img_res = []
        for m, src_m in enumerate(srcs):
        #    print(m)
            result = check_contours(m, dirname, img_name, src_m)

            if len(result) > 2:
                print(img_name ,'HELP')

            fake_result = [[0, [0, 0], [0, 0], [0, 0], 0]] * 2
            for j in range(min(len(result), 2)):
                fake_result[j] = result[j]

            result = fake_result[0]
            result += fake_result[1]

   #         print(result)
            img_res += result

        global_res.append(img_res + [img_name])

        if i % 1000 == 0:
           print(img_name, time.clock() - start)

    with open('3_mice_{}_table.csv'.format('_'.join(dirname.split('/'))), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(global_res)


if __name__ == '__main__':
    #print('Hey')
    tracking(sys.argv[1])