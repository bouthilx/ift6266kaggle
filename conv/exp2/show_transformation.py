import sys
import os
import ast

template = """!obj:contestTransformerDataset.TransformerDataset {
    raw : !obj:contest_dataset.ContestDataset {
            which_set: 'train',
            start: 0,
            stop: 4000,
    },
    transformer : %(transformation)s,
    space_preserving : True,
}"""

DIR = "/u/bouthilx/projects/ift6266kaggle/conv/exp2/"

t_yaml = sys.argv[1]

t_name = t_yaml.split(".yaml")[0]

t_id = sys.argv[2]

index = open(DIR+t_name+"/yaml/test.index","r").readlines()
# Ugliest command ever
#params = dict(ast.literal_eval(filter(lambda a:a.split(".yaml")[0][-len(t_id):]==t_id and not a.split(".yaml")[0][-(len(t_id)+1):].isdigit(), index)[0].split("==")[1].strip("\n").strip("\t").strip(" ")))
try: 
    print "try"
    params = dict(ast.literal_eval(filter(lambda a:a.split(".yaml")[0][-len(t_id):]==t_id and not a.split(".yaml")[0][-(len(t_id)+1):].isdigit(), index)[0].split("==")[1].strip("\n").strip("\t").strip(" ")))

except ValueError as e: 
    print "huh"
    if str(e)=="dictionary update sequence element #0 has length 1; 2 is required":
        params = dict((ast.literal_eval(filter(lambda a:a.split(".yaml")[0][-len(t_id):]==t_id and not a.split(".yaml")[0][-(len(t_id)+1):].isdigit(), index)[0].split("==")[1].strip("\n").strip("\t").strip(" ")),))
    else:
        raise e

print "Params",params

template = template % {"transformation":"".join(open(DIR+t_name+"/"+t_yaml,'r'))}
template = template % params

template_path = DIR+t_name+"/yaml/dataset_template"+t_id+".yaml"

t_file = open(template_path,"w")
t_file.write(template)
t_file.close()

os.system("python ~/pylearn2/pylearn2/scripts/show_examples.py --out=%s %s" % (template_path.replace(".yaml",".png"), template_path))
