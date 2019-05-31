import os, re, time
from upload import upload


cur_dir = ''
while True:
    glob_dirs = [d for d in os.listdir() if 'AM' in d or 'PM' in d]
    glob_dirs.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[0]),
                                  int(x.split('_')[1]), x.split('_')[5],
                                  int(x.split('_')[3]) % 12, int(x.split('_')[4])))

    if cur_dir != glob_dirs[-1]:
        cur_dir = glob_dirs[-1]
        os.system('ssh root@159.89.232.110 mkdir /root/Novosad/mouses/Data/{}'.format(cur_dir))
        os.system('ssh root@159.89.232.110 mkdir /root/Novosad/mouses/check/{}'.format(cur_dir))
        os.system('ssh root@159.89.232.110 mkdir /mnt/volume-nyc1-01/{}'.format(cur_dir))

        print(cur_dir)
        while not os.path.isfile(os.path.join(cur_dir, 'done.txt')):
            to_upload = [d for d in os.listdir(cur_dir) if 'AM' in d or 'PM' in d]
            to_upload.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[0]),
                                          int(x.split('_')[1]), x.split('_')[5],
                                          int(x.split('_')[3]) % 12, int(x.split('_')[4])))

            if len(to_upload) < 2:
                time.sleep(30)
                continue

            to_upload = to_upload[-2]
            upload(cur_dir, to_upload)

            time.sleep(30)

        to_upload = [d for d in os.listdir(cur_dir) if 'AM' in d or 'PM' in d]
        to_upload.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[0]),
                                      int(x.split('_')[1]), x.split('_')[5],
                                      int(x.split('_')[3]), int(x.split('_')[4])))

        to_upload = to_upload[-1]

        upload(cur_dir, to_upload)

        os.system('ssh root@159.89.232.110 python3 /root/Novosad/mouses/start_script.py {} {}'.format(cur_dir, to_upload))

    else:
        time.sleep(30)


