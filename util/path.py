import os
cur_dir = os.curdir
filepath = 'hoge/foo/hoge.txt'
print(cur_dir)
print(os.path.abspath(cur_dir))
print(__file__)
print(os.path.abspath(cur_dir))
base_name = os.path.basename(filepath)
print(base_name)
splited = os.path.splitext(base_name)
print(splited)
path, basename = os.path.split(filepath)
print(path, basename)