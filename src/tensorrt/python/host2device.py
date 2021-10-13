import json
import shlex
import argparse
import itertools
import subprocess
import typing as t
from dlbs.utils import LogEvent


"""
Example configuration file (JSON). GPU format: [CPU_CORES:]GPU (cpu cores are optional, is used for numactl).
```json
{
    "gpu": ["0-9:0", "10-10:1"],
    "size_mb": [19, 38, 75, 151, 302, 604, 1208],
    "pinned_mem": true,
    "num_warmup_iterations": 10,
    "num_iterations": 500,
    "docker": "nvidia-docker",
    "image": "dlbs/tensorrt:21.08"
}
```
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=False, default=None,
        help="Path to a JSON configuration file. If given, all other parameters are ignored."
    )
    parser.add_argument(
        '--docker', type=str, required=False, default='nvidia-docker',
        help="Docker executable."
    )
    parser.add_argument(
        '--image', type=str, required=False, default='dlbs/tensorrt:21.08',
        help="Docker image."
    )
    parser.add_argument(
        '--gpu', type=str, required=False, default="0",
        help="GPU index to use. The format is '[cpu_affinity:]GPU_ID', where cpu_affinity is the range of cores. If "
             "present, pin process to these range of CPUs with numactl and enforce local memory allocation policy. "
             "For instance, on a two socket NUMA machine with 18 cores per CPU, setting --gpu = '0-17:0' will "
             "effectively pin process to socket #0."
    )
    parser.add_argument(
        '--size_mb', type=float, required=False, default=10.0,
        help="Size of a data chunk in MegaBytes. During inference benchmarks, data is transferred as arrays of shape"
             "[BatchSize, 3, Wight, Height] of 'float' data type. These are typical sizes for AlexNetOWT where\n"
             "Width = Height = 227:\n"
             "\tBatch size (images):  32  64  128  256  512  1024\n"
             "\tBatch size (MB):      19  38   75  151  302   604\n"
    )
    parser.add_argument(
        '--pinned_mem', '--pinned-mem', required=False, default=False, action='store_true',
        help="Allocate buffer in host pinned memory."
    )
    parser.add_argument(
        '--num_warmup_iterations', '--num-warmup-iterations', type=int, required=False, default=10,
        help="Number of warmup iterations."
    )
    parser.add_argument(
        '--num_iterations', '--num-iterations', type=int, required=False, default=100,
        help="Number of benchmark iterations."
    )
    return parser.parse_args()


def run(docker: t.Text, image: t.Text, pinned_mem: bool, cpus: t.Optional[t.Text], gpu: t.Text,
        num_warmup_iterations: int, num_iterations: int, size_mb: float) -> t.Optional[float]:
    docker_cmd: t.Text = f"{docker} run -ti --rm"
    benchmark_cmd: t.Text = ""

    if cpus:
        docker_cmd += " --privileged"
        benchmark_cmd += f" numactl --localalloc --physcpubind={cpus}"

    benchmark_cmd += f" benchmark_host2device_copy --gpu={gpu} --size={size_mb} --num_batches={num_iterations}"\
                     f" --num_warmup_batches={num_warmup_iterations}"
    if pinned_mem:
        benchmark_cmd += " --pinned"

    throughput_mb_s: t.Optional[float] = None
    docker_cmd += f" {image} /bin/bash -c '{benchmark_cmd}'"
    with subprocess.Popen(shlex.split(docker_cmd), universal_newlines=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, encoding='utf-8') as p:
        while True:
            output = p.stdout.readline()
            if output == '' and p.poll() is not None:
                break
            if output:
                dlbs_event = LogEvent.from_string(output)
                if dlbs_event is not None and 'host2device' in dlbs_event.labels:
                    throughput_mb_s = dlbs_event.content['throughput_mb_s']
                else:
                    # sys.stdout.write("LINE '" + output.strip(' \t\n') + "'\n")
                    # sys.stdout.flush()
                    ...
    return throughput_mb_s


def main():
    args: argparse.Namespace = parse_arguments()
    if args.config:
        with open(args.config) as f:
            global_config = json.load(f)
    else:
        global_config = vars(args)
        _ = global_config.pop('config')

    for param in global_config:
        if not isinstance(global_config[param], list):
            global_config[param] = [global_config[param]]

    params, values = zip(*global_config.items())
    configs = (dict(zip(params, v)) for v in itertools.product(*values))
    for config in configs:
        config['cpus'] = None
        cpus_gpu: t.List[t.Text] = config['gpu'].split(':')
        if len(cpus_gpu) == 2:
            config['cpus'] = cpus_gpu[0]
            config['gpu'] = cpus_gpu[1]
        config['throughput_mb_s'] = run(**config)
        LogEvent(config, labels=['host2device']).log()


if __name__ == '__main__':
    main()
