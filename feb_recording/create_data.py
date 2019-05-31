import os
import ast

import numpy as np
import csv


def create():
    with open('/storage/Data/ucsf_concatenated.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))

    for m in range(4, 8):
        print(m)
        to_csv = []

        xls_ind = 0
        for _ in range(len(data)):
            if _ % 10000 == 0:
                print(m, _)

            try:
                data[_][m] = ast.literal_eval(data[_][m])
            except:
                pass
            data[_][m][0] = float(data[_][m][0])
            data[_][m][1] = [float(d) for d in data[_][m][1]]
            data[_][m][2] = [float(d) for d in data[_][m][2]]
            data[_][m][3] = [float(d) for d in data[_][m][3]]
            data[_][m][-1] = float(data[_][m][-1])

            if _ != len(data) - 1:
                data[_ + 1][m] = ast.literal_eval(data[_ + 1][m])
                data[_ + 1][m][0] = float(data[_ + 1][m][0])
                data[_ + 1][m][1] = [float(d) for d in data[_ + 1][m][1]]
                data[_ + 1][m][2] = [float(d) for d in data[_ + 1][m][2]]
                data[_ + 1][m][3] = [float(d) for d in data[_ + 1][m][3]]
                data[_ + 1][m][-1] = float(data[_ + 1][m][-1])

            if _ == 0:
                body_speed = 0
                head_speed = 0
                speed_vect = [1, 0]
                prev_head_body_dist = 10
                prev_head_body_vect = [1, 0]
            else:
                body_speed = np.sum((np.array(data[_][m][2]) - np.array(
                    data[_ - 1][m][2])) ** 2)
                body_speed = np.sqrt(body_speed)

                head_speed = np.sum((np.array(data[_][m][1]) - np.array(
                    data[_ - 1][m][1])) ** 2)
                head_speed = np.sqrt(head_speed)

                speed_vect = np.array(data[_][m][2]) - np.array(data[_ - 1][m][2])

            if _ != len(data) - 1:
                next_body_speed = np.sum((np.array(data[_ + 1][m][2]) - np.array(
                        data[_][m][2])) ** 2)
                next_body_speed = np.sqrt(next_body_speed)

                next_head_speed = np.sum((np.array(data[_ + 1][m][1]) - np.array(
                        data[_][m][1])) ** 2)
                next_head_speed = np.sqrt(next_head_speed)
            else:
                next_body_speed = 0
                next_head_speed = 0

            # acceleration
            body_acceleration = next_body_speed - body_speed
            head_acceleration = next_head_speed - head_speed

            # mean coordinates (percent of cage size)
            body_coord = data[_][m][2]
            # if m == 0:
            #     body_coord[0] = float(body_coord[0]) / 270  # left cage
            # if m == 1:
            #     body_coord[0] = float(body_coord[0] - 270) / 270  # right cage

            head_coord = data[_][m][1]
            # if m == 0:
            #     head_coord[0] = float(head_coord[0]) / 270  # left cage
            # if m == 1:
            #     head_coord[0] = float(head_coord[0] - 270) / 270  # right cage

            head_body_vect = np.array(data[_][m][1]) - np.array(data[_][m][2])
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
            temp = data[_][m][-1]

            # TABLE

            to_csv.append([data[_][1],
                           data[_][2] + data[_][3],
                           head_body_dist,
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
                           ])
            prev_head_body_dist = head_body_dist
            prev_head_body_vect = head_body_vect[:]

            xls_ind += 1

        with open(os.path.join('/storage/Data/', '{}_ucsf.csv'.format(m - 3)), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(to_csv)


if __name__ == '__main__':
    create()