import os
import sys, time


def start(dirname, subdirname):
    print('Start start')
    dirs = [d for d in os.listdir(os.path.join('Data', dirname, subdirname))]
    for dir in dirs:
        print('Started ', dirname, subdirname, dir)
        os.system('python3 tracking_w_head.py {} &'.format(os.path.join(dirname, subdirname, dir)))
    print('Start finish')


if __name__ == '__main__':
    start(sys.argv[1], sys.argv[2])
