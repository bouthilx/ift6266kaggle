from scipy import ndimage, misc
import pylab as pl
import numpy as np
import sys
import os
from collections import OrderedDict

import getopt
#from contest_dataset import ContestDataset

import __init__ as t

_verbose = False

def error():
    print """Try `python transform.py --help` for more information"""
    sys.exit(2)

def show_help():
    print """
Usage: python transform.py TRANSFORMATIONS DATASET NEW_DATASET 
Extend a dataset by adding random transformations

Transformations file configuration:

    Transformations
        #name settings
        translate ratio=5   mean=1    std=1   
        zoom      ratio=5   mean=1    std=0.1   
        rotate    ratio=5   mean=0    std=10    
        noise     ratio=0.5 sigma=1
        noise     ratio=0.5 sigma=2
        sharpen   ratio=0.5 sigma1=3  sigma2=1  alpha=30  
        denoise   ratio=0.5 sigma1=2  sigma2=3  alpha=0.4 
        clutter   ratio=5   mean=25   std=10.0  max_nb_per_image=2 
        flip      ratio=1

Example:
    python transformation.py -t transform.conf dataset.csv extended.csv
"""

def convert_params(name,value):
    if name in ['mean','std','ratio','alpha','sigma','sigma1','sigma2']:
        value = float(value)
    elif name in ['nb','max_nb_per_image']:
        value = int(value)
    else:
        raise BaseException("Unknown key : \"%s\"" % name)

    return (name,value)

def parse_transform(arg):
    # parse

    ts = OrderedDict()

    f = open(arg,'r')
    for line in f:
        if line[0].strip(" ")[0]=="#" or line.strip(" ").strip("\n")==" ":
            continue

        line = line.replace("\t"," ")

        params = filter(lambda a: a.strip(" ")!="",line.strip("\n").split(" "))
        params = [a.strip(" ") for a in params]
        if _verbose: print params
        name = params[0]
        if name not in t.transformations_dict.keys():
            raise BaseException("Invalid transformation : %s" % name)

        try:
            ts[name] = dict([convert_params(*a.split("=")) for a in params[1:]])
        except BaseException as e:
            raise BaseException(str(e)+" in transformation : %s" % name)

    return ts

def main(argv):
    save = ""
    force = False

    try:
        opts, args = getopt.getopt(argv,"vhf",
                        ["verbose","help","force"])
    except getopt.GetoptError as getopt_error:
        print getopt_error.msg, getopt_error.opt
        error()
    else:
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                show_help()
                sys.exit()
            elif opt in ("-v","--verbose"):
                global _verbose
                _verbose = True
            elif opt in ("-f","--force"):
                force = True

    transformations, load_path, save_path = read_args(args)

    if os.path.exists(save_path) and not force:
        print """file \"%(file)s\" already exists. 
    Use --force option if you wish to overwrite them""" % {"file":save_path}
        error()

    X, y = t.load_dataset(load_path)
    X, y = t.apply_transformations(X[:100],y[:100],transformations)
#    t.save_dataset(X,y,save_path)
    t.show_samples(X,y,100)

def read_args(args):
    if len(args)!=3:
        error()

    try:
        transformations = parse_transform(args[0])
    except BaseException as e:
        print e
        error()

    load_path = args[1]
    save_path = args[2]

    return transformations, load_path, save_path

if __name__ == "__main__":
    main(sys.argv[1:])
