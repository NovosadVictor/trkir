import os, sys, time
import shutil

from part_dir import part
from unzip import unzip
from start_script import start

N_THREADS = 6

os.chdir('/home/viktor/mice/new_recording')

import os
os.putenv('DISPLAY', ':0.0')

def script(dirname, subdirname):

    unzip(dirname, subdirname)

    part(dirname, subdirname)
    start(dirname, subdirname)

if __name__ == '__main__':
    script(sys.argv[1], sys.argv[2])