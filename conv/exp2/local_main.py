from pylearn2.utils.shell import run_shell_command
from pylearn2.utils import serial
from gen_yaml import generate_params, write_files

import contest_dataset

import os

DIR = "/home/xavier/ift6266kaggle/conv/exp2/"

OUT = "/yaml/test.yaml"
TEMPLATE = DIR+"template.yaml"
HPARAMS = "hparams.conf"

if __name__ == "__main__":

#    for transformation in ['translate','scale','rotate','flip','gaussian','sharpen','denoize','occlusion','halfface']:
#    for transformation in ['scale','rotate','flip','gaussian','sharpen','denoize','occlusion','halfface']:
    for transformation in ['denoize','sharpen']:
        out = DIR+transformation+OUT
        t_template = "".join(open(DIR+transformation+"/"+transformation+".yaml",'r'))

        # Generates a list of hyper-parameter names and a list of 
        # hyper-parameter values
        hpnames, hpvalues = generate_params(hparamfile=DIR+transformation+"/"+transformation+".conf",
                                            generate="log-uniform",
                                            search_mode="fix-grid-search")

        template = "".join(open(TEMPLATE,'r')) % {'transformations': t_template,'save_path':'%(save_path)s'}

        # Writes template with each hyper-parameter settings in  
        # succesive files and returns the name of the files
        files = write_files(template=template,hpnames=hpnames,
                            hpvalues=hpvalues,save_path=out,force=True)

#    files = write_files(template="".join(open(TEMPLATE),'r'),hpnames=hpnames,
#                        hpvalues=hpvalues,save_path=OUT)

        for f in files:
            serial.load_train_file(f).main_loop()
