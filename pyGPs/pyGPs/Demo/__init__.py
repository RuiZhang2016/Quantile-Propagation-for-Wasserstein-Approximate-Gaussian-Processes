import sys
id = [p for p,c in enumerate(__file__) if c == '/']
proj_path = __file__[:id[-2]]

print('add project path: ', proj_path)
sys.path.append(proj_path)

gp_path = proj_path+'/pyGPs'
print('add pyGPs path: ', gp_path)
sys.path.append(gp_path)

# for Mac OS
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')