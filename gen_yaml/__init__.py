#!/bin/env python

#Copyright (c) 2013, Xavier Bouthillier
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the <organization> nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



__authors__ = "Xavier Bouthillier" 
__contact__ = "xavier.bouthillier@umontreal.ca"

import sys
import os
import re
import numpy as np
from collections import OrderedDict

__all__ = ["generate_params","write_files"]

def error():
    print """Try `python gen_yaml.py --help` for more information"""
    sys.exit(2)

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


        if len(s_line)!=6:
            print "Incorrect hyper-parameter configuration"
            print line.strip("\n")
            print "# Hyper-parameters :: min :: max :: how much :: generation-mode :: default value"
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
    """
        Generate
    """

    hparams = OrderedDict()
    hpnbs = []

    for hparam in HparamReader(hparamfile):
        
        hpnbs.append(hparam['hpnb'])

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        hparam.pop('default')

#        hparam['hpnb'] = max(hpnb,hparam['hpnb'])

        if "random" not in hparam["generate"]:
            print "*** Warning ***"
            print "    Hyperparameter",hparam["hparam"],": Random search, Generation function =", generate
            print "    Random search but not a random value generation? Are you sure that's what you want?"

        name = hparam.pop("hparam")
        hparams[name] = hparams.get(name,[]) + list(make_hparams(**hparam))

    rand = []
    while len(rand) < min(hpnbs):
        r = int(np.random.rand(1)*min(hpnbs))
        if r not in rand:
            rand.append(r)

    rand = np.array(rand)/(.0+min(hpnbs))

    values = [np.array(hparam)[list(rand*len(hparam))] for hparam in hparams.values()]

    return hparams.keys(), np.transpose(np.array(values))

def fixgridsearch(hparamfile,generate):

    hparams = OrderedDict()
    dhparams = OrderedDict()

    for hparam in HparamReader(hparamfile):

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        dhparams[hparam['hparam']] = hparam.pop("default")

        name = hparam.pop("hparam")
        hparams[name] = hparams.get(name,[]) + list(make_hparams(**hparam))

    values = np.zeros((sum([len(hparam) for hparam in hparams.values()]),len(hparams.keys())))

    j = 0
    for i, hparam in enumerate(hparams.items()):
        # set all default values
        values[j:j+len(hparam[1])] = np.array(dhparams.values())
        # set the value of the current hyper-parameter
        values[j:j+len(hparam[1]),i] = np.array(hparam[1])

        j += len(hparam[1])

    return hparams.keys(), values

# http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
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

    hparams = OrderedDict()
    dhparams = OrderedDict()

    for hparam in HparamReader(hparamfile):

        if "generate" not in hparam or hparam["generate"] in ["default",""]:
            if hparam["generate"]=="":
                print "*** Warning ***"
                print "    Hyperparameter",hparam["hparam"]
                print "    Please set generation mode : default"

            hparam["generate"] = generate

        hparam.pop("default")

        name = hparam.pop("hparam")

        hparams[name] = hparams.get(name,[]) + list(make_hparams(**hparam))

    return hparams.keys(), cartesian(hparams.values())
 

search_modes = {"random-search":randomsearch,
		"fix-grid-search":fixgridsearch,
		"full-grid-search":fullgridsearch}

def write_files(template,hpnames,hpvalues,save_path,force=False):
#    template = "".join(open(template,'r'))
    save_path = re.sub('.yaml$','',save_path)

    files = []

    if hpvalues.shape[0]>40:
        a = ""
        while a not in ["y","n"]:
            a = raw_input("Do you realy want to produce as much as %d yaml files? (y/n) " % hpvalues.shape[0])
            if a=='n':
                sys.exit(0)


    # save templates
    for i, hparams in enumerate(hpvalues):
        file_name = '%(save_path)s%(i)d' % {"save_path":save_path,"i":i}
        
        d = dict(zip(hpnames,hparams))

        d.update({'save_path':file_name})

        try:
            tmp_template = template % d
        except KeyError as e:
            print "The key %(e)s is not present in hyper-parameter file" % {'e':e}
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

    f = open(save_path+".index",'w')
    f.write('\n'.join(files)+'\n')
    f.close()

    return [f.split(" == ")[0] for f in files]


def generate_params(hparamfile,generate,search_mode):
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
