# __Reporting__

DLBS provides basic functionality to build benchmarking reports. This is a two step
process: (1) parse log files and generate JSON files and (2) use JSON files to
provide higher level information such as strong/weak scaling reports etc.

### Log parser
Log [parser](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/dlbs/logparser.py)
is the first tool one needs to use to build reports. Log parser finds experiment log files
and simply extracts key-value pairs assuming that values are JSON parseable strings.
According to definition, a key-value pair in any textual file is a line that satisfies the following regular expression:
```
[ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)
```
The examples are:
```
__exp.num_batches__=100
__tensorflow.distortions__=false
__caffe.action__="train"
__results.use.cpu__=[580.0, 100.0, 100.0, 93.3]
```
Thus, every log file (which is the same as every experiment) is parsed into a JSON
object with keys being experiment parameters and values being their values. Lines
that do not match are ignored by the log parser. Log parser creates a JSON object
with single key named `data` which value is a list of json objects - experiment
parameters.

Assuming the environmental variable `parser` points to a log parser (i.e. `export parser=$DLBS/python/dlbs/logparser.py`), in the simplest case, log parser can parse
single log file, extract all parameters and print them out to a console:
```bash
python $parser ./bvlc_caffe/alexnet_2.log
```

Users can specify which fields (parameters) they are interested in with `--output_params`
command line argument:
```bash
python $parser ./bvlc_caffe/alexnet_2.log --output_params "exp.framework_title,exp.model_title,exp.effective_batch,results.time"
```

It is possible to specify multiple files:
```bash
python $parser ./bvlc_caffe/*.log --output_params "exp.framework_title,exp.model_title,exp.effective_batch,results.time"
```

It's also possible to specify a directory. In case of
directory, a switch `--recursive` can be used to find log files in that directory and all its
subdirectories:
```bash
python $parser ./bvlc_caffe --recursive --output_params "exp.framework_title,exp.model_title,exp.effective_batch,results.time"
```

To be able to process results and build reports, parameters need to be serialized to a json
file:
```bash
params="exp.framework_title,exp.model_title,exp.effective_batch,results.time"
python $parser ./bvlc_caffe --recursive --output_file ./bvlc_caffe.json --output_params ${params}
```

### Summary Builder
[Summary Builder](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/dlbs/summary_builder.py) builds simple exploration or weak/strong-scaling reports based on JSON files produced
by the log parser.
