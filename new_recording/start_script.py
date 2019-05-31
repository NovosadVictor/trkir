import os
import sys, time

os.putenv('DISPLAY', ':0.0')

def start(dirname, subdirname):
    print('Start start')
    dirs = [d for d in os.listdir(os.path.join('data', dirname, subdirname))]
    for dir in dirs:
        print('Started ', dirname, subdirname, dir)
        os.system('python3.6 new_tracking.py {} &'.format(os.path.join(dirname, subdirname, dir)))
    print('Start finish')


if __name__ == '__main__':
    start(sys.argv[1], sys.argv[2])
