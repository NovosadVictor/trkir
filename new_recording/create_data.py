import os
import sys

import xlrd, xlwt
import numpy as np
import csv


def create():
    xlses = []

    for xls in xlses:
        print(xls)
        mouses_info = [[[] for i in range(8)] for j in range(8)]

        dates = [[], []]

        with open(os.path.join('csvs', xls), newline='') as csvfile:
            data = list(csv.reader(csvfile))[1:]

            print(data[:10])
            if len(data) > 20:
                data = np.array(data).T.tolist()

        mouses_info = [[data[8 * m + i] for i in range(8)] for m in range(8)]
        dates = data[-2:]

        for m in range(8):
            mouses_info[m] = np.array(mouses_info[m]).T.astype(float).tolist()

        for m in range(8):
            # sheet1 = xlwt.Workbook()
            # table = sheet1.add_sheet('data from mouse %d' % m)

            to_csv = []

            xls_ind = 0
            for _ in range(len(mouses_info[m])):
                if _ == 0:
                    body_speed = 0
                    head_speed = 0
                    speed_vect = [1, 0]
                    prev_head_body_dist = 10
                    prev_head_body_vect = [1, 0]
                else:
                    body_speed = np.sum((np.array([mouses_info[m][_][0], mouses_info[m][_][1]]) - np.array(
                        [mouses_info[m][_ - 1][0], mouses_info[m][_ - 1][1]])) ** 2)
                    body_speed = np.sqrt(body_speed)

                    head_speed = np.sum((np.array([mouses_info[m][_][2], mouses_info[m][_][3]]) - np.array(
                        [mouses_info[m][_ - 1][2], mouses_info[m][_ - 1][3]])) ** 2)
                    head_speed = np.sqrt(head_speed)

                    speed_vect = np.array([mouses_info[m][_][0], mouses_info[m][_][1]]) - np.array([mouses_info[m][_ - 1][0], mouses_info[m][_ - 1][1]])

                if _ != len(mouses_info[m]) - 1:
                    next_body_speed = np.sum((np.array([mouses_info[m][_ + 1][0], mouses_info[m][_ + 1][1]]) - np.array(
                            [mouses_info[m][_][0], mouses_info[m][_][1]])) ** 2)
                    next_body_speed = np.sqrt(next_body_speed)

                    next_head_speed = np.sum((np.array([mouses_info[m][_ + 1][2], mouses_info[m][_ + 1][3]]) - np.array(
                            [mouses_info[m][_][2], mouses_info[m][_][3]])) ** 2)
                    next_head_speed = np.sqrt(next_head_speed)
                else:
                    next_body_speed = 0
                    next_head_speed = 0

                # acceleration
                body_acceleration = next_body_speed - body_speed
                head_acceleration = next_head_speed - head_speed

                # mean coordinates (percent of cage size)
                body_coord = [mouses_info[m][_][0], mouses_info[m][_][1] / 480]
                # if m == 0:
                #     body_coord[0] = float(body_coord[0]) / 270  # left cage
                # if m == 1:
                #     body_coord[0] = float(body_coord[0] - 270) / 270  # right cage

                head_coord = [mouses_info[m][_][2], mouses_info[m][_][3] / 480]
                # if m == 0:
                #     head_coord[0] = float(head_coord[0]) / 270  # left cage
                # if m == 1:
                #     head_coord[0] = float(head_coord[0] - 270) / 270  # right cage

                head_body_vect = np.array([mouses_info[m][_][2], mouses_info[m][_][3]]) - np.array([mouses_info[m][_][0], mouses_info[m][_][1]])
                head_body_dist = np.linalg.norm(head_body_vect)
                head_body_dist_change = head_body_dist - prev_head_body_dist

                if np.linalg.norm(speed_vect) != 0:
                    cos = np.dot(head_body_vect, speed_vect) / (np.linalg.norm(speed_vect) * head_body_dist)

                    if np.isnan(cos):
                        cos = 1

                    sin = np.sqrt(1 - cos * cos)

                else:
                    cos = 1
                    sin = 0
                body_speed_x = body_speed * cos
                body_speed_y = body_speed * sin

                head_speed_x = head_speed * cos
                head_speed_y = head_speed * sin

                if head_body_dist != 0 and prev_head_body_dist != 0:
                    if np.array_equal(head_body_vect, prev_head_body_vect):
                        head_rotation = 0
                    else:
                        head_rotation = np.arccos(np.dot(head_body_vect, prev_head_body_vect) / (head_body_dist * prev_head_body_dist))

                    if np.isnan(head_rotation):
                        head_rotation = 0
                else:
                    head_rotation = 0


                # avg temperature
                temp = mouses_info[m][_][6]

                # time

           #     print(dates[1])
                h, minutes, s = (dates[1][_].split()[1]).split(':')
                h, minutes, s = int(h), int(minutes), int(s)
                if 'PM' in dates[1][_] and h != 12:
                    h += 12
                if h == 12 and 'AM' in dates[1][_]:
                    h = 0

                time = h * 3600 + minutes * 60 + s

                # TABLE

                # SHAPE
                # table.write(xls_ind, 0, head_body_dist)
                #
                # # MOTION
                # table.write(xls_ind, 1, body_speed_x)
                # table.write(xls_ind, 2, body_speed_y)
                # table.write(xls_ind, 3, head_speed_x)
                # table.write(xls_ind, 4, head_speed_y)
                # table.write(xls_ind, 5, head_body_dist_change)
                # table.write(xls_ind, 6, head_rotation)
                # table.write(xls_ind, 7, body_speed)
                # table.write(xls_ind, 8, head_speed)
                # table.write(xls_ind, 9, body_acceleration)
                # table.write(xls_ind, 10, head_acceleration)
                #
                # # WINDOW
                # table.write(xls_ind, 11, body_coord[0])
                # table.write(xls_ind, 12, body_coord[1])
                # table.write(xls_ind, 13, head_coord[0])
                # table.write(xls_ind, 14, head_coord[1])
                #
                # # OTHER
                # table.write(xls_ind, 15, temp)
                # table.write(xls_ind, 16, time)
                #
                # # DATES
                # table.write(xls_ind, 17, dates[0][_])
                # table.write(xls_ind, 18, dates[1][_])

                to_csv.append([head_body_dist,
                               body_speed_x,
                               body_speed_y,
                               head_speed_x,
                               head_speed_y,
                               head_body_dist_change,
                               head_rotation,
                               body_speed,
                               head_speed,
                               body_acceleration,
                               head_acceleration,
                               body_coord[0],
                               body_coord[1],
                               head_coord[0],
                               head_coord[1],
                               temp,
               #                time,
                               dates[0][_],
                               dates[1][_]])
                prev_head_body_dist = head_body_dist
                prev_head_body_vect = head_body_vect[:]

                xls_ind += 1

            with open(os.path.join('data_csv', '{}_{}'.format(m, xls)), 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(to_csv)
            #sheet1.save(os.path.join('check', dirname, 'xlses', 'data', '{}_{}.xls'.format(m, subdirname)))


if __name__ == '__main__':
    create()