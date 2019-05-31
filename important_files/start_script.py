import os, sys, time
import shutil

from part_dir import part
from script import start
from clear_xlses import clear
from add_empty import add_empt
from create_data import create
from kstest import test
from concatenate import concatenate
from unzip import unzip

N_THREADS = 6

os.chdir('/root/Novosad/mouses/')


def script(dirname, subdirname):

    unzip(dirname, subdirname)

    if not os.path.isdir('check/{}'.format(dirname)):
        os.mkdir('check/{}'.format(dirname))

    if not os.path.isdir('/mnt/volume-nyc1-01/{}'.format(dirname)):
        os.mkdir('/mnt/volume-nyc1-01/{}'.format(dirname))

    if not os.path.isdir(os.path.join('check', dirname, 'xlses')):
        os.mkdir(os.path.join('check', dirname, 'xlses'))

    if not os.path.isdir(os.path.join('check', dirname, 'xlses', subdirname)):
        os.mkdir(os.path.join('check', dirname, 'xlses', subdirname))

    try:
        if os.path.isfile(os.path.join('/root/Novosad/mouses/Data', dirname, subdirname + '.zip')):
            ext = '.zip'
        if os.path.isfile(os.path.join('/root/Novosad/mouses/Data', dirname, subdirname + '.rar')):
            ext = '.rar'

        shutil.move(os.path.join('/root/Novosad/mouses/Data', dirname, subdirname + ext), os.path.join('/mnt/volume-nyc1-01/', dirname))
    except Exception as e:
        print(e, 'already moved', subdirname)

    part(dirname, subdirname)
    start(dirname, subdirname)

    time.sleep(20)

    try:
        while True:
            if len([f for f in os.listdir(os.path.join('check', dirname, subdirname)) if 'done.txt' in f]) == N_THREADS:
                xls_files = []

                for numb in range(N_THREADS):
                    xls_files += [os.path.join('check', dirname, subdirname, str(numb), f)
                                  for f in os.listdir(os.path.join('check', dirname, subdirname, str(numb))) if '.xls' in f]

                for xls in xls_files:
                    shutil.move(xls, os.path.join('check', dirname, 'xlses', subdirname))

                break

            else:
                time.sleep(60 * 1)
    except Exception as e:
        print(e)

    clear(dirname, subdirname)
    add_empt(dirname, subdirname)

    create(dirname, subdirname)

    concatenate('0', dirname, subdirname)
    concatenate('1', dirname, subdirname)

    infected = []
    if os.path.isdir(os.path.join('check', dirname, 'xlses', 'avg', '144')):
        infected = [f for f in os.listdir(os.path.join('check', dirname, 'xlses', 'avg', '144')) if '.xls' in f]
    infected = list(set([f[0] for f in infected]))

    for m in infected:
        m = '0' if m == 'l' else '1'
        test(m, dirname)




if __name__ == '__main__':
    script(sys.argv[1], sys.argv[2])