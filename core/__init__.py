import sys
import os

# for Mac OS
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
else:
    pass
ROOT_PATH = os.path.split(os.getcwd())[0]
