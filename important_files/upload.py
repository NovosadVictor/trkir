import os
import sys
import shutil
import subprocess


def upload(dirname, subdirname):
    print('Trying to upload ', dirname, subdirname)
    if not os.path.isfile(os.path.join(dirname, 'zips', subdirname + '.zip')):
        shutil.make_archive(os.path.join(dirname, 'zips', subdirname), 'zip', dirname, subdirname)
    else:
        return

    if os.path.isfile(os.path.join(dirname, 'zips', subdirname + '.zip')):
        print('scping ', dirname, subdirname)
        p = subprocess.Popen(["scp", '{}/zips/{}.zip'.format(dirname, subdirname),
                              "root@159.89.232.110:/root/Novosad/mouses/Data/{}/".format(dirname)])
        sts = p.wait()


if __name__ == '__main__':
    upload(sys.argv[1], sys.argv[2])