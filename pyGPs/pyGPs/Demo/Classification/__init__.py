import sys
import os

# for Mac OS
if sys.platform == 'darwin':
    import matplotlib

    matplotlib.use('TkAgg')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    os.environ['proj'] = '/home/rzhang/PycharmProjects/WGPC'
sys.path.append(os.environ['proj'] + '/pyGPs')
sys.path.append(os.environ['proj'])