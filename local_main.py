from pylearn2.utils.shell import run_shell_command
from pylearn2.utils import serial
from gen_yaml import generate_params, write_files

import contest_dataset

import sys
import os

DIR = "/home/xavier/ift6266kaggle/conv/exp3/"

OUT = DIR+"yaml/test.yaml"
TEMPLATE = DIR+"template.yaml"
HPARAMS = DIR+"hparams.conf"

if __name__ == "__main__":

    # Generates a list of hyper-parameter names and a list of 
    # hyper-parameter values
    hpnames, hpvalues = generate_params(hparamfile=HPARAMS,
                                        generate="uniform",
                                        search_mode="fix-grid-search")

    force = len(sys.argv)>1 and sys.argv[1]=="--force"

    # Writes template with each hyper-parameter settings in  
    # succesive files and returns the name of the files
    files = write_files(template="".join(open(TEMPLATE,"r")),hpnames=hpnames,
                        hpvalues=hpvalues,save_path=OUT,force=force)

    for f in files:
        serial.load_train_file(f).main_loop()
#    for i in range(46-24):
#        f = DIR+"yaml/second%d.yaml" % (i+24)
#        print i+24,"on",46,"done"
#        serial.load_train_file(f).main_loop()
