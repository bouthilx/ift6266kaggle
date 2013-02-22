from pylearn2.utils.shell import run_shell_command
from pylearn2.utils import serial
from gen_yaml import generate_params, write_files

import contest_dataset

import os

DIR = "/home/xavier/ift6266kaggle/mlp/exp2/"

OUT = DIR+"yaml/second.yaml"
TEMPLATE = DIR+"template.yaml"
HPARAMS = DIR+"hparams.conf"

if __name__ == "__main__":

    # Generates a list of hyper-parameter names and a list of 
    # hyper-parameter values
    hpnames, hpvalues = generate_params(hparamfile=HPARAMS,
                                        generate="log-uniform",
                                        search_mode="fix-grid-search")

    # Writes template with each hyper-parameter settings in  
    # succesive files and returns the name of the files
    files = write_files(template=TEMPLATE,hpnames=hpnames,
                        hpvalues=hpvalues,save_path=OUT)

    for i, f in enumerate(files):
        print i,"on",len(files),"done"
        serial.load_train_file(f).main_loop()
