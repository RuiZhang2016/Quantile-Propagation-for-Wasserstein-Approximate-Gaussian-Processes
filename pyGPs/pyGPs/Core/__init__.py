from __future__ import absolute_import
from . import inf
from . import cov
from . import mean
from . import lik
from . import gp
from . import opt

# import sys
# id = [p for p,c in enumerate(__file__) if c == '/']
# proj_path = __file__[:id[-2]]

# print('add project path: ', proj_path)
# sys.path.append(proj_path)

# gp_path = proj_path+'/pyGPs'
# print('add pyGPs path: ', gp_path)
# sys.path.append(gp_path)

# for Mac OS
# if sys.platform == 'darwin':
#         import matplotlib
#         matplotlib.use('TkAgg')

__all__ = ['inf', 'cov', 'mean', 'lik','gp','opt']
