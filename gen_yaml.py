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
                                  (-np.log(hpmin)-np.log(hpmax))/(hpnb+.0)))),
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

def error():
    print """Try `python gen_yaml.py --help` for more information"""
    sys.exit(2)

def show_help():
    print """
Usage: python gen_yaml.py [OPTIONS] TEMPLATE-FILE H-PARAMETER-FILE
Produce yaml files given a template and hyper-parameters ranges
    -s FILE, --save=FILE    file name on which it builds yaml files {{1,2,3,...}}
                            default = TEMPLATE-FILE{{1,2,3,...}}.yaml
    -f, --force             Force yaml files creation even if files with the 
                            same name already exist
    -g, --generate=MODE     Generation mode. Applied to every learning rate.
                            Locally defined generation mode has predominance
                            {generation_modes}
                            
    -v, --verbose           Verbose mode

File configurations:

    Yaml template
        Use %(save_path)s for the filename of the yaml template. 
        The file name to save .pkl models in the yaml file will be the same as the 
        yaml file name. For a file test1.yaml, save_path will be replaced by test1.
        Look at the template.yaml file for an example.

    Hyper parameters configuration file
        # Hyper-parameters  min     max     how much (optional generation mode)

Example:
    python gen_yaml.py --save=mlp template.yaml hparams.conf
""".format(generation_modes=", ".join(generation_modes.keys()))

def main(argv):
    save = ""
    force = False
    generate = "log-uniform"

    try:
        opts, args = getopt.getopt(argv,"vhfs:g:",
                        ["verbose","help","force","save=","generate="])
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
            elif opt in ("-s","--save"):
                save = re.sub('.yaml$','',arg)
            elif opt in ("-g","--generate"):
                if arg not in ["log-random-uniform","log-uniform"]:
                    print "generate MODE is invalid: " +arg
                    error()
                generate = arg

    template, hparams = read_args(args)

    if not save:
        save = re.sub('.yaml$','',args[0])

    hyperparams = {}#('learning_rate',[0.01])]
    default_hparams = {}#{'learning_rate':0.1}

    # load hparams
    for line in hparams:
        if line.strip(" ")[0]=="#" or line.strip(" ").strip("\n")=="": 
            continue

        line = line.strip(" ").strip("\n")

        local_generate = ""

        s_line = filter(lambda a:a.strip(' ')!='',line.split(' '))
        s_line = [s.strip("\n").strip("\t") for s in s_line]
       
        if len(s_line)==5:
            hparam, hpmin, hpmax, hpnb, local_generate = s_line
        elif len(s_line)==4:
            hparam, hpmin, hpmax, hpnb = s_line
        else:
            print "Incorrect hyper-parameter configuration"
            print line.strip("\n")
            print "# Hyper-parameters  min     max     how much"
            error()

        hpmin, hpmax, hpnb = float(hpmin), float(hpmax), int(hpnb)
        hyperparams[hparam] = make_hparams(hpmin,hpmax,hpnb,
                                local_generate if local_generate else generate)
        if _verbose:
            print hparam, " : ", hpmin, hpmax, hpnb
            print "    ", hyperparams[hparam]

        if ((local_generate and local_generate[:3]=="log") or 
            (not local_generate and generate[:3]=="log")):
            default_hparams[hparam] = np.exp((np.log(hpmax)+np.log(hpmin))/2.0)
        else:
            default_hparams[hparam] = (hpmax+hpmin)/2.0



    # fill template
    template = ''.join(template)

    i = 1

    files = []

    # save templates
    for hparam, values in hyperparams.items():
        default_hparam = default_hparams[hparam]
        for value in values:
            file_name = '%(save_path)s%(i)d' % {"save_path":save,"i":i}

            default_hparams.update({hparam:value,'save_path':file_name})

            try:
                tmp_template = template % default_hparams
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
            
            default_hparams.pop('save_path')
            files.append("%(file_name)s == %(hparams)s" % {"file_name":file_name,
                         "hparams":', '.join([str(v) for v in default_hparams.items()])})

            i += 1

        default_hparams[hparam] = default_hparam

    f = open(save+".index",'w')
    f.write('\n'.join(files)+'\n')
    f.close()

    if _verbose:
        print '\n'.join(files)+'\n'

# open the file and set the tape string
def read_args(args):
    if len(args)>2:
        print "Too many arguments given: ",len(args)
        error()
    elif len(args)<2:
        print "Missing file arguments"
        error()
    else:
       return open(args[0],'r').readlines(), open(args[1],'r').readlines()

def make_hparams(hpmin,hpmax,hpnb,fct):
    try:
        return generation_modes[fct](hpmin,hpmax,hpnb)
    except KeyError as e:
        print "invalid generative function : ",fct
        print "Try ",", ".join(generation_modes.keys())
        error()

if __name__ == "__main__":
    main(sys.argv[1:])
