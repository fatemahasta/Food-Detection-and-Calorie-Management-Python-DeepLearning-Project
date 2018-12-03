import os
from collections import defaultdict

src = 'food-101/test'
path = 'food-101/meta/train.txt'
dir_files = defaultdict(list)
with open(path, 'r') as txt:
    files = [l.strip() for l in txt.readlines()]
    for f in files:
        d = os.path.join(src, (f + '.jpg'))
        os.remove(d)
        
src = 'food-101/train'
path = 'food-101/meta/test.txt'
dir_files = defaultdict(list)
with open(path, 'r') as txt:
    files = [l.strip() for l in txt.readlines()]
    for f in files:
        d = os.path.join(src, (f + '.jpg'))
        os.remove(d)
        







