import os

def make_savefolder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)