#!/bin/env python

__authors__ = "Xavier Bouthillier" 
__contact__ = "xavier.bouthillier@umontreal.ca"

import getopt
import sys
import os
import time
import re
import numpy as np
from collections import defaultdict

_verbose = False

generation_modes = {
    "log-uniform": 
        lambda hpmin, hpmax, hpnb :
            list(np.exp(np.arange(np.log(hpmin),np.log(hpmax),
                                  (np.log(hpmax)-np.log(hpmin))/(hpnb+.0)))),
    "log-random-uniform": 
        lambda hpmin, hpmax, hpnb : 
            list(np.exp(np.random.uniform(np.log(hpmin),
                                          np.log(hpmax),hpnb))),
    "uniform":
        lambda hpmin, hpmax, hpnb :
            list(np.arange(hpmin,hpmax,
                           (hpmax-hpmin)/(hpnb+.0))),
    "random-uniform":
        lambda hpmin, hpmax, hpnb : 
            list(np.random.uniform(hpmin,hpmax,hpnb)),
}

class HparamReader():
    def __init__(self,file_name):
        self.i = iter(filter(lambda a: a.strip(" ")[0]!="#" and a.strip(" ").strip("\n")!="",
                             open(file_name,'r').readlines()))

    def __iter__(self):
        return self

    def next(self):
        return self.build_hparam(self.i.next())

    def build_hparam(self,line):
        s_line = filter(lambda a:a.strip(' ')!='',line.split(' '))
        s_line = filter(lambda a:a.strip(' ')!='',[s.strip("\n").strip("\t") for s in s_line])


        if len(s_line)< 4 or len(s_line)>6:
            print "Incorrect hyper-parameter configuration"
            print line.strip("\n")
            print "# Hyper-parameters  min     max     how much"
            error()

        d = dict(zip(["hparam","hpmin","hpmax","hpnb","generate","default"],s_line))
        
        for h in ["hpmin","hpmax","default"]:
            if h in d.keys():
                d[h] = float(d[h])

        for h in ["hpnb"]:
            if h in d:
                d[h] = int(d[h])
 
        return d

def randomsearch(hparamfile,generate):

    hparams = []
    names = []
    hpnb = None

    for hparam in HparamReader(hparamfile):
        
        if not hpnb:
            hpnb = hparam['hpnb']

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        if 'default' in hparam:
            hparam.pop('default')

        hparam['hpnb'] = hpnb

        if "random" not in hparam["generate"]:
            print "*** Warning ***"
            print "    Hyperparameter",hparam["hparam"],": Random search, Generation function =", generate
            print "    Random search but not a random value generation? Are you sure that's what you want?"

        names.append(hparam.pop("hparam"))
        hparams.append(make_hparams(**hparam))

    values = np.zeros((sum([len(hparam) for hparam in hparams]),len(hparams)))

    return names, np.transpose(np.array(hparams))


def fixgridsearch(hparamfile,generate):

    hparams = []
    dhparams = []
    names = []

    for hparam in HparamReader(hparamfile):

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        default = None
        if "default" in hparam :
            default = hparam.pop("default")
            

        names.append(hparam.pop("hparam"))
        hparams.append(make_hparams(**hparam))

        if not default:
            default = hparams[-1][int(hparam['hpnb'])/2]

        dhparams.append(default)



    values = np.zeros((sum([len(hparam) for hparam in hparams]),len(hparams)))

    j = 0
    for i, hparam in enumerate(hparams):
        values[j:j+len(hparam)] = np.array(dhparams)
        values[j:j+len(hparam),i] = np.array(hparam)

        j += len(hparam)

    return names, values

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
    1-D arrays to form the cartesian product of.
    out : ndarray
    Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
    2-D array of shape (M, len(arrays)) containing cartesian products
    formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def fullgridsearch(hparamfile,generate):
    # load hparams
    hparams = []
    names = []

    for hparam in HparamReader(hparamfile):

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        if 'default' in hparam:
            hparam.pop("default")

        names.append(hparam.pop("hparam"))
        hparams.append(make_hparams(**hparam))

    return names, cartesian(hparams)
 

search_modes = {"random-search":randomsearch,
		"fix-grid-search":fixgridsearch,
		"full-grid-search":fullgridsearch}

def error():
    print """Try `python gen_yaml.py --help` for more information"""
    sys.exit(2)

def show_help():
    print """
Usage: python gen_yaml.py [OPTIONS] TEMPLATE-FILE H-PARAMETER-FILE
Produce yaml files given a template and hyper-parameters ranges
    -o FILE, --out=FILE    file name on which it builds yaml files {{1,2,3,...}}
                           default = TEMPLATE-FILE{{1,2,3,...}}.yaml
    -f, --force            Force yaml files creation even if files with th 
                           same name already exist
    -s, --search=MODE	   Search mode. 
		           {search_modes}
                           default : fix-grid-search
    -g, --generate=MODE    Generation mode. Applied to every learning rate.
                           Locally defined generation mode has predominance
                           default, {generation_modes}
                           default : log-uniform
                            
    -v, --verbose          Verbose mode

File configurations:

    Yaml template
        Use %(save_path)s for the filename of the yaml template. 
        The file name to save .pkl models in the yaml file will be the same as the 
        yaml file name. For a file test1.yaml, save_path will be replaced by test1.
        Look at the template.yaml file for an example.

    Hyper parameters configuration file
        # Hyper-parameters  min     max     how much (optional generation mode)

Example:
    python gen_yaml.py --out=mlp template.yaml hparams.conf
""".format(generation_modes=", ".join(generation_modes.keys()),search_modes=", ".join(search_modes.keys()))

def main(argv):
    save = ""
    force = False
    generate = "log-uniform"
    search_mode = "fix-grid-search"

    try:
        opts, args = getopt.getopt(argv,"vhfo:s:g:",
                        ["verbose","help","force","out=","search=","generate="])
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
            elif opt in ("-o","--out"):
                save = re.sub('.yaml$','',arg)
            elif opt in ("-g","--generate"):
                if arg not in generation_modes.keys():
                    print "generate MODE is invalid: " +arg
                    error()
                generate = arg
            elif opt in ("-s","--search"):
                if arg not in search_modes.keys():
                    print "search MODE is invalid: " +arg
                    error()
                search_mode = arg



    template, hparams = read_args(args)

    if not save:
        save = re.sub('.yaml$','',args[0])

    hpnames, hpvalues = make_search(hparams,generate,search_mode)

    # fill template
    template = ''.join(template)

    i = 1

    files = []

    if hpvalues.shape[0]>40:
        a = ""
        while a not in ["y","n"]:
            a = raw_input("Do you realy want to produce as much as %d yaml files? (y/n) " % hpvalues.shape[0])
            if a=='n':
                sys.exit(0)


    # save templates
    for i, hparams in enumerate(hpvalues):
        file_name = '%(save_path)s%(i)d' % {"save_path":save,"i":i}

        d = dict(zip(hpnames,hparams))

        d.update({'save_path':file_name})

        try:
            tmp_template = template % d
        except KeyError as e:
            print "The key %(e)s is not present in %(hparamsfile)s" % {'e':e,'hparamsfile':args[1]}
            error()

        file_name += '.yaml'

        if os.path.exists(file_name) and not force:
            print """file \"%(file)s\" already exists. 
Use --force option if you wish to overwrite them""" % {"file":file_name}
            error()
        else:
            f = open(file_name,'w')
            f.write(tmp_template)
            f.close()
        
        d.pop('save_path')
        files.append("%(file_name)s == %(hparams)s" % {"file_name":file_name,
                     "hparams":', '.join([str(v) for v in d.items()])})

    f = open(save+".index",'w')
    f.write('\n'.join(files)+'\n')
    f.close()

    if _verbose:
        print '\n'.join(files)+'\n'

def read_args(args):
    if len(args)>2:
        print "Too many arguments given: ",len(args)
        error()
    elif len(args)<2:
        print "Missing file arguments"
        error()
    else:
       return args[0], args[1]

def make_search(hparamfile,generate,search_mode):
    try:
        return search_modes[search_mode](hparamfile,generate)
    except KeyError as e:
        print "invalid search function : ",search_mode
        print "Try ",", ".join(search_modes.keys())
        error()

def make_hparams(hpmin,hpmax,hpnb,generate):
    try:
        return generation_modes[generate](hpmin,hpmax,hpnb)
    except KeyError as e:
        print "invalid generative function : ",generate
        print "Try ",", ".join(generation_modes.keys())
        error()

if __name__ == "__main__":
    main(sys.argv[1:])
