import sys
import os

# for Mac OS
if sys.platform == 'darwin':
    import matplotlib

    matplotlib.use('TkAgg')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    os.environ['proj'] = '/home/users/u5963436/Work/WGPC'# '/home/rzhang/PycharmProjects/WGPC' # '/home/users/u5963436/Work/WGPC'
sys.path.append(os.environ['proj'] + '/pyGPs')
sys.path.append(os.environ['proj'])
