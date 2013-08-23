Copyright (c) 2013, Xavier Bouthillier
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


==============================
gen_yaml : Yaml file generator
==============================

Produce yaml files given a template and hyper-parameters ranges.
It also writes an index file (save_path).index with all the file names and
hyper-parameters settings.

Usage
****
  python gen_yaml.py [OPTIONS] TEMPLATE-FILE H-PARAMETER-FILE

::

  -o FILE, --out=FILE    file name on which it builds yaml files {1,2,3,...}
                         default = TEMPLATE-FILE{1,2,3,...}.yaml
  -f, --force            Force yaml files creation even if files with th 
                         same name already exist
  -s, --search=MODE      Search mode. 
                         default : fix-grid-search
                         fix-grid-search :  Vary hyper-parameters one at a 
                                            time, keeping the others to a 
                                            default value.
                         full-grid-search : Compute all possible combinations
                                            of hyper-parameters values that
                                            has been generated
                         random-search :    Generate random values for every
                                            hyper-parameter given the range
                                            specified. Should be used with
                                            a random generation mode, a warning
                                            message will show up otherwise.
  -g, --generate=MODE    Generation mode. Applied to every hyper-parameter with 
                         default value. Locally defined generation mode has predominance
                         default, random-uniform, log-uniform, log-random-uniform, uniform
                         default : log-uniform                        
  -v, --verbose          Verbose mode

File configurations
****

Yaml template
        Use %(save_path)s for the save_path of the yaml template. 
        The save_path to save the .pkl models in the yaml template file will be 
        the same as the yaml file name. 
        For a file test1.yaml, save_path will be replaced by test1.
        Take a look at the template.yaml file for an example.

Hyper parameters configuration file
        # Hyper-parameters  : min : max : how much : generate mode : default value (optional) 
        
        Take a look at the hparams.conf file for an example.

Examples
********

Run from file
=============

    python gen_yaml.py --out=yaml/test.yaml template.yaml hparams.conf

Run from code
=============
.. code-block:: python

  from gen_yaml import generate_params, write_files

  # Generates a list of hyper-parameter names and a list of hyper-parameter values
  hpnames, hpvalues = generate_params(hparamfile="hparams.conf",generate="log-uniform",
                                      search_mode="fix-grid-search")

  # Writes template with each hyper-parameter settings in succesiv files 
  # and returns the name of the files
  files = write_files(template="template.yaml",hpnames=hpnames,
                      hpvalues=hpvalues,save_path="yaml/test.yaml")

  for f in files:
      print f

.. code-block:: python
