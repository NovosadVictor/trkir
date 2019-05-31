"""
Get [area, head_coord, center_coord, tail_coord, temp_value] for image with mouse
"""
import os
import time
import sys
import csv


import cv2 as cv
import numpy as np
import matplotlib.cm as cm


PIX_TRH = 0.55
MIN_AREA = 1.5 * 1e2
MAX_AREA = 3 * 1e3
G_BLUR = (15, 15)
COLORS = [color * 255 for color in cm.rainbow(np.linspace(0, 1, 3))]  # head, center, tail


def detect_mouse_data_from_url(img_url):
    img = cv.imread(img_url)

    return detect_mouse_data_from_img(img)


def detect_mouse_data_from_img(img):
    gray_img, black_white_img = get_gray_and_black_white_img(img)

    white_contours = get_white_contours(black_white_img)

    contours_data = get_contours_data(white_contours)

    possible_body_parts = get_possible_body_parts_data_from_contours(contours_data, gray_img, black_white_img)

    mouse_data = get_head_tail_temp(possible_body_parts)

    return mouse_data


def get_gray_and_black_white_img(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blurred_img = cv.GaussianBlur(gray_img, G_BLUR, 0)

    trsh_const = np.max([PIX_TRH * np.max(img), 100])
    _, black_white_img = cv.threshold(blurred_img, trsh_const, 255, cv.THRESH_BINARY)

    return gray_img, black_white_img


def get_white_contours(black_white_img):
    _, white_contours, _ = cv.findContours(black_white_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    return white_contours


def get_contours_data(contours):
    contour_results = []

    for c in contours:
        area = cv.contourArea(c)

        if area < MIN_AREA or MAX_AREA < area:
            continue

        center, eigenvect = get_orientation(c)
        contour_results.append([area, center, eigenvect, c])

    return contour_results


def get_orientation(contour):
    # SOME SHIT
    # !TODO: fix shit
    contour_len = len(contour)
    data_contour = np.empty((contour_len, 2), dtype=np.float64)
    for i in range(data_contour.shape[0]):
        data_contour[i, 0] = contour[i, 0, 0]
        data_contour[i, 1] = contour[i, 0, 1]

    mean = np.empty(0)
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_contour, mean)

    center = (int(mean[0, 0]), int(mean[0, 1]))

    # eigenvectors[0] is the longest(most distribution = body alignment) eigenvector
    return center, eigenvectors[0]


def get_possible_body_parts_data_from_contours(contours_data, gray_img, black_white_img):
    possible_body_parts = []

    for cur_values in contours_data:
        area, center, eigenvect, contour = cur_values

        temp, temp_arg = get_temperature_for_contour(contour, gray_img)
        head_or_tail = get_head_tail_points_for_contour(black_white_img, center, eigenvect)

        head, tail = decide_head_tail(head_or_tail, temp_arg)

        possible_body_parts.append([float(area), list(head), list(center), list(tail), float(temp)])

    return possible_body_parts


def get_temperature_for_contour(contour, gray_img):
    # draw contour on black image
    black_img = np.zeros_like(gray_img)
    cv.drawContours(black_img, [contour], -1, color=255, thickness=-1)

    # !TODO: fix (extra black img?)
    # we need to leave only one white part from gray image
    white_contour_indexes = np.where(black_img == 255)
    black_img[white_contour_indexes] = gray_img[white_contour_indexes]

    temp_arg = np.unravel_index(black_img.argmax(), black_img.shape)
    temp = gray_img[temp_arg]

    return temp, temp_arg


def get_head_tail_points_for_contour(black_white_img, center, eigenvect):
    # we need to find intersection of pca eigenvector and mouse's body contour (= head + tail)
    # SOME SHIT
    # !TODO: fix shit

    # we start from center
    head_or_tail = [center[:], center[:]]

    eigenvect = set_vector_direction(eigenvect)
    # go to opposite direction along the eigenvector and find intersection
    try:
        while black_white_img[int(head_or_tail[0][1]), int(head_or_tail[0][0])] != 0:
            head_or_tail[0] = [head_or_tail[0][0] + eigenvect[0], head_or_tail[0][1] + eigenvect[1]]

        while black_white_img[int(head_or_tail[1][1]), int(head_or_tail[1][0])] != 0:
            head_or_tail[1] = [head_or_tail[1][0] - eigenvect[0], head_or_tail[1][1] - eigenvect[1]]
    except:
        pass

    return head_or_tail


def set_vector_direction(eigenvect):
    # set eigenvector direction to "up"
    if eigenvect[0] * eigenvect[1] > 0:
        eigenvect = [abs(eigenvect[0]), abs(eigenvect[1])]
    else:
        if eigenvect[0] > 0:
            eigenvect = [-eigenvect[0], -eigenvect[1]]

    # if eigenvector is close to vertical
    if eigenvect[1] / eigenvect[0] > 50:
        eigenvect = [0, 1]

    # if eigenvector is close to horizontal
    elif eigenvect[0] / eigenvect[1] > 50:
        eigenvect = [1, 0]

    return eigenvect


def decide_head_tail(head_or_tail, temp_arg):
    # eyes is the hottest part of mouses body, so head should be closer to temp_arg
    if np.linalg.norm(np.array(head_or_tail[0]) - np.array(temp_arg)) < np.linalg.norm(np.array(head_or_tail[1]) - np.array(temp_arg)):
        head, tail = head_or_tail
    else:
        head, tail = head_or_tail[::-1]

    return head, tail


def get_head_tail_temp(possible_body_parts):
    return possible_body_parts[0]


def draw_body_parts(img=None, img_url='', drawing_parts=None, save_url='', show=False):
    if img_url:
        img_to_draw = cv.imread(img_url)
    if img is not None:
        img_to_draw = img[:]

    drawing_parts = drawing_parts or detect_mouse_data_from_img(img_to_draw)

    list_ind_for_color = 0

    for part_ind, drawing_part in enumerate(drawing_parts):
        for part in drawing_part:
            if isinstance(part, list):
                cv.circle(img_to_draw, (int(part[0]), int(part[1])), radius=5, color=COLORS[list_ind_for_color], thickness=3)
                list_ind_for_color += 1
                list_ind_for_color %= 3

    if show:
        cv.imshow('image', img_to_draw)

    if save_url:
        cv.imwrite(save_url, img_to_draw)

    return img_to_draw


def draw_text(img, text, bias_ind,
              font=cv.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255)):

    text_bottom_left_corner = (100, bias_ind * 30)

    cv.putText(img, text,
               text_bottom_left_corner,
               font,
               font_scale,
               font_color)


def pix_temp_to_temp(stat_file, temp_pixes):
    with open(stat_file, 'r') as stat_f:
        stat_data = stat_f.readlines()

    for i in range(len(temp_pixes)):
        splitted_line = stat_data[i].split()
        min_temp = float(splitted_line[-2])
        max_temp = float(splitted_line[-1])

        for j in range(len(temp_pixes[i])):
            temp_pixes[i][j] = (min_temp * (255 - temp_pixes[i][j]) + max_temp * temp_pixes[i][j]) / 255


def detect_for_dir(dirname, pc):
    print(dirname)
    urls = [os.path.join(dirname, 'JPEG', url) for url in os.listdir(os.path.join(dirname, 'JPEG'))]
    urls.sort(key=lambda url: int(url.split('/')[-1].split('.')[0]))

    save_url = '/storage/Data/new_{}_processed/'.format(pc)
    strange_url = '/storage/strange_detected/'
    cage_centers = {'pc': [358, 238], 'ucsf': [330, 240]}
    cage_center = cage_centers[pc]

    full_data = []
    for url_i, url in enumerate(urls):
        if url_i % int(0.3 * len(urls)) == 0:
            print(url)

        img = cv.imread(url)

        mouse_data = []
        for m in range(4):
            if m == 0:
                crop_img = img[: cage_center[1], : cage_center[0]]
            if m == 1:
                crop_img = img[: cage_center[1], cage_center[0]:]
            if m == 2:
                crop_img = img[cage_center[1]:, : cage_center[0]]
            if m == 3:
                crop_img = img[cage_center[1]:, cage_center[0]:]

            try:
                mouse_data.append(detect_mouse_data_from_img(crop_img))
            except:
                mouse_data.append([])

            time.sleep(0.001)

            # if url_i % 1000 == 0:
            #     draw_body_parts(img=crop_img, drawing_parts=[mouse_data[-1]], save_url=strange_url + str(m) + '_' + url.split('/')[-1])

        full_data.append(mouse_data)
        time.sleep(0.002)

    with open(save_url + '{}.csv'.format(dirname.split('/')[-1]), 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(full_data)


if __name__ == '__main__':
    detect_for_dir(sys.argv[1], sys.argv[2])
