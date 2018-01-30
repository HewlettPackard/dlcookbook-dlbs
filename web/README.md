# __Experiment progress__

We provide a very basic and simple tool to monitor progress of benchmark experiments.
To enable this functionality, run experimenter with `--progress-file` command line
argument providing a full path to a file that will be used to store progress. This
is a JSON file that will be updated by an experimenter.
```bash
python experimenter --progress-file=/dev/shm/progress.json ...
```
You can view this file and track progress. Another possibility is to run a simple
web server. This will enable tracking progress with your web browser. The web server
implementation is located in `python/dlbs/web`:
```
python ./simple_server.py /dev/shm/progress.json 8000
```
You need to provide a file path to a JSON file that (the one that was used to run
experimenter, and a port number). 
