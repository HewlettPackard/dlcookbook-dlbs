### Advanced Configuration IN PROGRESSSSSSSS

Basically, one benchmark is fully specified by its parameters such as per device batch size, model name, framework, log file etc. The goal of experimenter is to take some input configuration and generate one or multiple benchmark experiments. Input configuration consists of multiple sections each defining different aspects of how benchmarks are generated. These sections are the following:


```json
{
    'parameters':{},
    'variables': {},
    'extensions':[
        {
            'condition':{},
            'parameters': {},
            'cases': [
                {}
            ]
        }
    ]
}
```

Experimenter uses the following algorithm to build plan of experiments (see [Builder](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/dlbs/builder.py) class for implementation details):

```python
1.  plan <- []
2.  for exp_vars in CartesianProduct(variables):
5.      experiments <- [copy(parameters).update(exp_vars)]
6.      for extension in extensions:
7.          active_experiments <- []
8.          for experiment in experiments:
8.              if not match(experiment, extension.condition):
9.                  active_experiments.append(experiment)
10.             active_experiments.extend( extend(extended_config, extension) )
11.        experiments <- active_experiments
12.     plan <- plan.extend(experiment)
```


1. Parameters
2. Variables
3. Extensions:
    1. Conditions
    2. Parameters
    3. Cases

It should be clear after reading this page how experimenter generate benchmark experiments.
