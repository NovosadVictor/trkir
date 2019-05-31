import os
import time
import sys
import gc
import re
import resource

import numpy as np
from PIL import Image, ImageDraw
import xlwt

from sklearn.cluster import AgglomerativeClustering


DATA_DIR = './Data/'
TEST_DIR = './check/'
dirname = sys.argv[1]

X_bias_l, X_bias_r = -70, 0

model_l = AgglomerativeClustering(n_clusters=2)
model_r = AgglomerativeClustering(n_clusters=2)

print('START')
imgs = [f for f in os.listdir(os.path.join(DATA_DIR, dirname, 'JPEG'))
        if os.path.isfile(os.path.join(DATA_DIR, dirname, 'JPEG', f))]
imgs.sort(key=lambda x: int(x.split('.')[0]))

last = imgs[0]

qq = imgs.index(last)

imgs = imgs[imgs.index(last):]

therm_file = os.listdir(os.path.join(DATA_DIR, dirname, 'Stats'))[0]

sheet = xlwt.Workbook()
table = sheet.add_sheet('data from ' + dirname.split('/')[-1])

date_style = xlwt.XFStyle()
date_style.num_format_str = 'M/D/YY'

time_style = xlwt.XFStyle()
time_style.num_format_str = 'h:mm:ss'

ind_pre = 0
with open(os.path.join(DATA_DIR, dirname, 'Stats', therm_file)) as therm_f:
    therm = therm_f.readlines()


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 5, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def write_to_table(_, centers, heads, temps, splitted, file):
    table.write(_, 0, int(centers[0][1]))
    table.write(_, 1, int(centers[0][0]))
    table.write(_, 2, int(heads[0][1]))
    table.write(_, 3, int(heads[0][0]))
    table.write(_, 4, 0)
    table.write(_, 5, 0)

    table.write(_, 6, temps[0])

    table.write(_, 7, int(centers[1][1]))
    table.write(_, 8, int(centers[1][0]))
    table.write(_, 9, int(heads[1][1]))
    table.write(_, 10, int(heads[1][0]))
    table.write(_, 11, 0)
    table.write(_, 12, 0)

    table.write(_, 13, temps[1])

    table.write(_, 14, file)  # filename

    table.write(_, 15, splitted[1], date_style)  # date
    table.write(_, 16, splitted[2], time_style)  # time
    table.write(_, 17, splitted[3])  # AM / PM


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


def tracking(dirname, file, min_temp, max_temp, prev_temps):
    img = np.array(Image.open(os.path.join(DATA_DIR, dirname, 'JPEG', file)).convert('RGB'))
    shape = img.shape
    pic = img[:, :, 0].copy()

    left_tr = list(set(np.sort(pic.T[: (shape[1] - X_bias_l) // 2], axis=None)))
    right_tr = list(set(np.sort(pic.T[(shape[1] - X_bias_l) // 2: shape[1] - X_bias_r], axis=None)))

    white_part_left = np.where(pic >= left_tr[int(0.6 * len(left_tr))])
    white_part_right = np.where(pic >= right_tr[int(0.6 * len(right_tr))])

    head_part_left = np.where(pic >= left_tr[-1] - 1)
    head_part_right = np.where(pic >= right_tr[-1] - 1)

    boundaries = [[0, (img.shape[1] - X_bias_l) // 2, (img.shape[1] - X_bias_l) // 2, img.shape[1] - X_bias_r],
                  [0, 480, 0, 480]]
    white_parts = split_parts(white_part_left, white_part_right, boundaries)

    head_parts = split_parts(head_part_left, head_part_right, boundaries)

    # predict_l = model_l.fit_predict(np.array(white_parts[0]).T)
    #
    # mouses_l = [(np.array(white_parts[0]).T[np.where(predict_l == 0)][:, 0],
    #              np.array(white_parts[0]).T[np.where(predict_l == 0)][:, 1]),
    #             (np.array(white_parts[0]).T[np.where(predict_l == 1)][:, 0],
    #              np.array(white_parts[0]).T[np.where(predict_l == 1)][:, 1])]
    #
    # c_l_1 = np.mean(mouses_l[0], axis=1)
    # c_l_2 = np.mean(mouses_l[1], axis=1)
    #
    # is_found = True
    # if np.linalg.norm(c_l_1 - c_l_2) > 40:
    #     is_found = False
    #
    # if not is_found:
    #     dist_1 = np.linalg.norm(c_l_1 - np.array([340, 135]))
    #     dist_2 = np.linalg.norm(c_l_2 - np.array([340, 135]))
    #     white_parts[0] = mouses_l[0] if dist_1 > dist_2 else mouses_l[1]
    #
    # predict_r = model_r.fit_predict(np.array(white_parts[1]).T)
    #
    # mouses_r = [(np.array(white_parts[1]).T[np.where(predict_r == 0)][:, 0],
    #              np.array(white_parts[1]).T[np.where(predict_r == 0)][:, 1]),
    #             (np.array(white_parts[1]).T[np.where(predict_r == 1)][:, 0],
    #              np.array(white_parts[1]).T[np.where(predict_r == 1)][:, 1])]
    #
    # c_r_1 = np.mean(mouses_r[0], axis=1)
    # c_r_2 = np.mean(mouses_r[1], axis=1)
    #
    # is_found = True
    # if np.linalg.norm(c_r_1 - c_r_2) > 40:
    #     is_found = False
    #
    # if not is_found:
    #     dist_1 = np.linalg.norm(c_r_1 - np.array([340, 345]))
    #     dist_2 = np.linalg.norm(c_r_2 - np.array([340, 345]))
    #     white_parts[1] = mouses_r[0] if dist_1 > dist_2 else mouses_r[1]

    centers = [np.mean(white_parts[0], axis=1), np.mean(white_parts[1], axis=1)]

    left_h = np.mean(head_parts[0], axis=1)
    right_h = np.mean(head_parts[1], axis=1)

    temps = [0, 0]
    for j in range(2):
        temps[j] = (min_temp * (255 - np.max(pic[white_parts[j]])) + max_temp * np.max(
            pic[white_parts[j]])) / 255
        if prev_temps[j] is not None:
            if temps[j] < 28.0 and abs(prev_temps[j] - temps[j]) < 1.0:
                pass
            else:
                if not (600 < len(white_parts[j][0]) < 1500):
                    temps[j] = prev_temps[j]
                else:
                    if temps[j] >= 30.5:
                        pass
                    elif prev_temps[j] - temps[j] > 1.0:
                        temps[j] = prev_temps[j]

    # img[white_parts[0]] = [temps[0] * 5, 0, 0]
    # img[white_parts[1]] = [0, 0, temps[1] * 5]
    #
    # img = Image.fromarray(img, mode='RGB')
    # draw = ImageDraw.Draw(img)
    #
    # # MEAN CENTER
    # draw.ellipse([centers[0][1] - 3, centers[0][0] - 3, centers[0][1] + 3, centers[0][0] + 3], fill=(100, 255, 255))
    # draw.ellipse([left_h[1] - 3, left_h[0] - 3, left_h[1] + 3, left_h[0] + 3], fill=(100, 255, 255))
    # draw.ellipse([centers[1][1] - 3, centers[1][0] - 3, centers[1][1] + 3, centers[1][0] + 3], fill=(100, 255, 255))
    # draw.ellipse([right_h[1] - 3, right_h[0] - 3, right_h[1] + 3, right_h[0] + 3], fill=(100, 255, 255))
    # draw.line([boundaries[0][1], boundaries[1][0], boundaries[0][1], boundaries[1][1]], fill=(255, 255, 255))
    # img.save(os.path.join(DATA_DIR, 'to_check', file))

    return centers, [left_h, right_h], temps


def detect_body():
    memory_limit()

    prev_temps = [None, None]
    if os.path.isfile(os.path.join(TEST_DIR, dirname, 'last_saved.txt')):
        last_saved = open(os.path.join(TEST_DIR, dirname, 'last_saved.txt'), 'r')
        last = last_saved.readlines()
        if len(last) > 1:
            prev_centers = [(int(last[1]), int(last[2])), (int(last[3]), int(last[4]))]
            prev_temps = [float(last[5]), float(last[6])]
        last = last[0][:-1]
        last_saved.close()
    string_ind = int(re.split('[._-]', imgs[0])[-2])

    for _, file in enumerate(imgs):
        start = time.clock()
        try:
            if _ % int(0.1 * len(imgs)) == 0:
                print(_, len(imgs), file)

            start_time = time.clock()

            splitted = therm[string_ind].split()
            min_temp = float(splitted[-2])
            max_temp = float(splitted[-1])

            centers, heads, temps = tracking(dirname, file, min_temp, max_temp, prev_temps)

            end_time = time.clock()

            write_to_table(_, centers, heads, temps, splitted, file)

            prev_centers = centers
            prev_temps = temps

            if _ % int(0.1 * len(imgs)) == 0:
                print(file, end_time - start_time, 'per image')

            string_ind += 1

            if _ > 0 and _ % 999 == 0:
                with open(os.path.join(TEST_DIR, dirname, 'last_saved.txt'), 'w') as f:
                    f.write(
                        file + '\n' + str(int(prev_centers[0][0])) + '\n' + str(int(prev_centers[0][1])) + '\n' + str(
                            int(prev_centers[1][0])) + '\n' + str(int(prev_centers[1][1])) + '\n' + str(
                            prev_temps[0]) + '\n' + str(prev_temps[1]))
        except Exception as e:
            print(e, file)
            gc.collect()

        print(time.clock() - start)

    sheet.save(os.path.join(TEST_DIR, dirname, '{}_{}.xls'.format(dirname.replace('/', '_'), qq)))
    print(imgs[-1], 'DONE\n\n\n\n\n')
    with open(os.path.join(TEST_DIR, '/'.join(dirname.split('/')[:-1]), '{}_done.txt'.format(dirname.split('/')[-1])), 'w') as f:
        f.write(dirname + ' done')


if __name__ == '__main__':

    if len(sys.argv) > 1:
        if not os.path.isdir(os.path.join(DATA_DIR, sys.argv[1])):
            print('There is no such directory ', sys.argv[1])
            sys.exit()

        if not os.path.isdir(os.path.join(TEST_DIR, sys.argv[1])):
            os.makedirs(os.path.join(TEST_DIR, sys.argv[1]))
    else:
        print('Write dirname')
        sys.exit()

    detect_body()