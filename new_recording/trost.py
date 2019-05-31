import os
import sys
import traceback
from tracking import detect_for_dir
import zipfile

path = '/storage/Data/last_23april_recording/04_19_19_10_36_AM_1'
if sys.argv[1] == 'pc':
    path = '/storage/Data/last_23april_recording/04_19_19_10_36_AM_2'

dirs = [os.path.join(path, z) for z in os.listdir(path) if '.zip' in z]

for d in dirs[:]:
    print(d)
    # if os.path.isfile('/storage/Data/new_{}_processed/{}.csv'.format(sys.argv[1], d.split('/')[-1][:-4])):
    #     continue
    try:
 #       with zipfile.ZipFile(d, "r") as zip_ref:
 #          zip_ref.extractall(path)
        detect_for_dir(d[:-4], sys.argv[1])
    except Exception as e:
        print(e)
        traceback.print_exc()
        print('Excepted ' + d)
        continue
