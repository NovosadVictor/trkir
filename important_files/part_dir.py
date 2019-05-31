import os
import re
import shutil
import sys
from math import ceil


DATA = 'Data'


def part(dirname, subdirname):
	try:
		print('Part start')
		stats = os.listdir(os.path.join(DATA, dirname, subdirname, 'Stats'))
		images = os.listdir(os.path.join(DATA, dirname, subdirname, 'JPEG'))
		images.sort(key=lambda x: (int(re.split('[.]', x)[-2])))

		parts = 6
		split = ceil(len(images) / parts)
		images = [images[i * split: (i + 1) * split] for i in range(parts)]

		for i in range(parts):
			os.mkdir(os.path.join(DATA, dirname, subdirname, str(i)))
			os.mkdir(os.path.join(DATA, dirname, subdirname, str(i), 'Stats'))
			os.mkdir(os.path.join(DATA, dirname, subdirname, str(i), 'JPEG'))

		for i in range(parts):
			for stat in stats:
				shutil.copy(os.path.join(DATA, dirname, subdirname, 'Stats', stat), os.path.join(DATA, dirname, subdirname, str(i), 'Stats'))
			for img in images[i]:
						shutil.copy(os.path.join(DATA, dirname, subdirname, 'JPEG', img), os.path.join(DATA, dirname, subdirname, str(i), 'JPEG'))

		os.system('rm -rf {}/{}/{}/JPEG'.format(DATA, dirname, subdirname))
		os.system('rm -rf {}/{}/{}/Stats'.format(DATA, dirname, subdirname))

		print('Part finish')
	except Exception as e:
		print(e)


if __name__ == '__main__':
	part(sys.argv[1], sys.argv[2])
	
