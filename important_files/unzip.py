import os, sys


def unzip(dirname, subdirname):
    print('Unzip start')

    if os.path.isfile('./Data/{0}/{1}.zip'.format(dirname, subdirname)):
        ext = 'zip'
    if os.path.isfile('./Data/{0}/{1}.rar'.format(dirname, subdirname)):
        ext = 'rar'

    if ext == 'zip':
        os.system('unzip ./Data/{0}/{1}.zip -d ./Data/{0}/'.format(dirname, subdirname))
    if ext == 'rar':
        os.system('unrar x ./Data/{0}/{1}.rar ./Data/{0}/'.format(dirname, subdirname))
    print('Unzip finish')


if __name__ == '__main__':
    unzip(sys.argv[1], sys.argv[2])