#!/usr/bin/env python3
import sys
import tempfile
import shlex
import os
import argparse
from string import Template
import json
import subprocess
import codecs

class myTemplate(Template):
    delimiter='!'
    def __init__(self,template):
        super().__init__(template)

def testforutf8():
    if sys.stdout.encoding.upper() != 'UTF-8':
        enc=sys.stdout.encoding.upper()
        raise ValueError(
           '''
           The encoding for sys.stdout is not UTF-8.
           Export PYTHONIOENCODING at the shell or set it on the command line.
           I.e.,
           export PYTHONIONENCODING=UTF-8
           or,
           PYTHONIONENCODING=UTF-8 {} ....
           '''.format(sys.argv[0]))

def main():
    try: testforutf8()
    except ValueError as e:
        print(e)
        sys.exit(-1)

    parser = argparse.ArgumentParser(description='Build a Singularity image from a recipe template file.')
    parser.add_argument('--recipe', '-r', type=str, required=True, help='Singularity recipe file or tempate file to process.')
    parser.add_argument('--macros','-m', dest='macros', type=str, action='store', default=None, help='Optional JSON file containing macro substitutions.')
    parser.add_argument('--image','-i', dest='image', type=str, action='store', default='image.img', help='Name of Singularity image to create.')
    args = parser.parse_args()

    if args.macros is not None:
        with open(args.recipe, 'r') as rf:
            try:
                with open(args.macros,'r') as mf: macros=json.load(mf)['macros']
                with tempfile.NamedTemporaryFile(mode='w',prefix='Singularity_',delete=False) as w:
                    for l in rf:
                        print(myTemplate(l.strip()).substitute(macros),file=w)
                    recipe_file=w.name
                print('Created temporary recipe file: {}'.format(recipe_file))
            except Exception as e:
                print('Build error. Was unable to process macro file {}.'.format(args.macros))
                print('Error: {} {}'.format(type(e),e))
                sys.exit(-1)
    else: recipe_file=args.recipe
    cmd="sudo /opt/singularity/bin/singularity build {} {}".format(args.image,recipe_file)
    process = subprocess.Popen(shlex.split(cmd), shell=False, bufsize=1,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ)
    while True:
        output= process.stdout.readline()
        pp=process.poll()
        if isinstance(output,bytes): output=output.decode('utf-8')
        if output == '' and (pp == 0 or pp is None): break
        if output !='':
           print(output.strip())

    if args.macros is not None:
        try:
            os.remove(recipe_file)
        except Exception:
            pass
 
if __name__=="__main__":
    main()
