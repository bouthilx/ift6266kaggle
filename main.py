from pylearn2.utils.shell import run_shell_command
from gen_yaml import generate_params, write_files

import os

DIR = "/data/lisatmp/ift6266h13/bouthilx/"

OUT = DIR+"yaml/test.yaml"
TEMPLATE = DIR+"gen_yaml/template.yaml"
HPARAMS = DIR+"gen_yaml/hparams.conf"

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

    command = """jobdispatch --condor --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=48:00:00 --whitespace --gpu bash %(dir)strain.py \"{{%(files)s\"}}""" % {"dir":DIR,"files":files[0]}
    output, rc = run_shell_command(command)
