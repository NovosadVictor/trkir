import os, xlrd, sys
import numpy as np
from PIL import Image, ImageDraw


def check_tracking(img_dir, xls_path=None):
    print(img_dir)
    if xls_path is None:
        xls_path = [os.path.join('check', img_dir, f) for f in os.listdir(os.path.join('check', img_dir)) if '.xls' in f][0]

    imgs = os.listdir(os.path.join('Data', img_dir, 'JPEG'))
    imgs.sort(key=lambda x: int(x.split('.')[0]))

    sheet = xlrd.open_workbook(xls_path)
    sheet = sheet.sheet_by_index(0)

    data = []
    dates = []
    for i in range(sheet.nrows):
        if '' in sheet.row_values(i):
            data.append(data[-1])
            dates.append(dates[-1])
        else:
            data.append(np.array(sheet.row_values(i)[:-4]).astype(float).tolist())
            dates.append(sheet.row_values(i)[-3:])

    if not os.path.isdir(os.path.join('Data', img_dir.split('/')[-1])):
        os.mkdir(os.path.join('Data', img_dir.split('/')[-1]))
    im_ind = 0

    for i in range(len(imgs)):
        if i % int(0.05 * len(imgs)) == 0:
            print(i, len(imgs))
        try:
            img = Image.open(os.path.join('Data', img_dir, 'JPEG', imgs[i])).convert('RGB')

            draw = ImageDraw.Draw(img)

            draw.text([270, 15], ' '.join(dates[i]))

            draw.text([20, 450], 'x: {} y: {}'.format(data[i][0], data[i][1]))
            draw.text([500, 450], 'x: {} y: {}'.format(data[i][7], data[i][8]))

            draw.ellipse([data[i][0] - 3, data[i][1] - 3, data[i][0] + 3, data[i][1] + 3], fill=(255, 0, 0))
            draw.ellipse([data[i][2] - 3, data[i][3] - 3, data[i][2] + 3, data[i][3] + 3], fill=(200, 0, 0))

            draw.ellipse([data[i][7] - 3, data[i][8] - 3, data[i][7] + 3, data[i][8] + 3], fill=(0, 0, 255))
            draw.ellipse([data[i][9] - 3, data[i][10] - 3, data[i][9] + 3, data[i][10] + 3], fill=(0, 0, 200))

            img.save(os.path.join('Data', img_dir.split('/')[-1], '%d.jpg' %im_ind))

            im_ind += 1

        except Exception as e:
            print(imgs[i], e)

    os.system('ffmpeg -framerate 12 -i Data/{}/%d.jpg Data/videos/{}.mp4'.format(img_dir.split('/')[-1], img_dir.split('/')[-1]))
    os.system('rm -rf Data/{}'.format(img_dir.split('/')[-1]))


if __name__ == '__main__':
    check_tracking(sys.argv[1], sys.argv[2])

