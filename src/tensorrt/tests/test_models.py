import os
import sys
import shutil
import mlflow
import logging
import tempfile
import typing as t
import subprocess
from dlbs.bench_data import BenchDataApp


logger = logging.getLogger()


class BenchmarkStatus:
    """ Holder for benchmark statuses. """
    FAILURE: int = 0
    SUCCESS: int = 1


def get_benchmark_status(benchmark: t.Union[str, t.Dict]) -> int:
    """ Return status ('success' or 'failure') of a benchmark. """
    if isinstance(benchmark, str):
        # If it's a string, consider this to be a path to a log file.
        benchmarks: t.List[t.Dict] = BenchDataApp(
            args=dict(
                action='parse', inputs=[benchmark], no_recursive=True, ignore_errors=False,
                select=None, update=None, output=None, report=None
            )
        ).load()
        if len(benchmarks) != 1:
            return BenchmarkStatus.FAILURE
        benchmark = benchmarks[0]

    try:
        if benchmark.get('exp.status', None) is not None:
            return benchmark.get('exp.status')
        throughput: float = float(benchmark.get('results.time', None))
        if throughput > 0:
            return BenchmarkStatus.SUCCESS
    except TypeError:
        ...
    return BenchmarkStatus.FAILURE


def get_tensorrt_exec() -> str:
    """ Return executable for DLBS TensorRT inference benchmarks. """
    tensorrt_exec: str = os.environ.get('DLBS_TENSORRT_EXEC', None)
    return tensorrt_exec or 'tensorrt'


def get_mlflow_artifact_uri() -> str:
    """ Return MLFlow artifact URI making sure it's a local file system. """
    artifact_uri: str = mlflow.get_artifact_uri()
    if not artifact_uri.startswith('file://'):
        raise ValueError(f"Unsupported artifact URI protocol (artifact_uri = {artifact_uri})")
    return artifact_uri[7:]


class CafeModels:
    """ Helper functions to work with DLBS Cafe models. """

    @staticmethod
    def get_model_title(file_path: str) -> t.Optional[str]:
        """ Open model file and find its title if present. """
        with open(file_path, 'r') as config:
            for line in config:
                if line.startswith('name:'):
                    return line[5:].strip()[1:-1]
        return None

    @staticmethod
    def init_cafe_file(model: t.Mapping, params: t.Mapping) -> t.Dict:
        """ Prepare Cafe training/inference file for a benchmark. """
        target_dir: str = params['path']
        phase: str = params['phase']
        target_file: str = model[phase]['alias'] + '.prototxt'
        batch_size = str(params['batch_size'])

        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, target_file)

        if phase == 'inference':
            with open(model[phase]['path'], 'r') as input_stream:
                with open(target_path, 'w') as output_stream:
                    for config_line in input_stream:
                        output_stream.write(config_line.replace('__EXP_DEVICE_BATCH__', batch_size))

        return dict(model_path=target_path, model_dir=target_dir, model_file=target_file, phase=phase)

    @staticmethod
    def get_models(path: str) -> t.List[t.Dict]:
        """ Get all supported models recursively. """
        models: t.List[t.Dict] = []
        for root, _, files in os.walk(path):
            for file in files:
                if not file.endswith(('.prototxt', '.onnx')):
                    continue
                model = dict(
                    model_path=os.path.join(root, file), model_dir=root, model_file=file,
                    name=file[0:-9], title=file[0:-9], type=os.path.splitext(file)[1][1:]
                )
                if model['type'] == 'prototxt':
                    model['title'] = CafeModels.get_model_title(model['model_path'])
                models.append(model)
        return models

    def __init__(self, source_dir: t.Optional[str] = None) -> None:
        if source_dir is None:
            source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models'))
        self.source_dir = source_dir

    def get_standard_models(self) -> t.List[t.Mapping]:
        """ Get information on all DLBS Cafe models. """
        models: t.List[t.Mapping] = []
        for model_dir in (d for d in os.scandir(self.source_dir) if d.is_dir()):
            model = dict(model_dir=model_dir.path, name=model_dir.name)
            for model_file in (f for f in os.scandir(model_dir.path) if f.name.endswith('.prototxt')):
                if model_file.name.endswith('.inference.prototxt'):
                    model['inference'] = dict(
                        path=model_file.path, file=model_file.name, alias=model_file.name[0:-19],
                        title=CafeModels.get_model_title(model_file.path)
                    )
                elif model_file.name.endswith('.training.prototxt'):
                    model['training'] = dict(
                        path=model_file.path, file=model_file.name,
                        alias=model_file.name[0:-18], title=CafeModels.get_model_title(model_file.path)
                    )
            models.append(model)
        return models


def test_dlbs_caffe_models() -> None:
    """ Test standard DLBS Caffe (prototxt) models. """
    params = dict(batch_size=1, gpus='0', dtype='float32', num_warmup_batches=1, num_batches=1,
                  tensorrt_exec=get_tensorrt_exec())
    models_dir = tempfile.mkdtemp()
    logger.info("Initializing standard DLBS Cafe models for TensorRT inference benchmarks (path=%s)", models_dir)
    for model in CafeModels().get_standard_models():
        _ = CafeModels.init_cafe_file(model, dict(path=models_dir, phase='inference',
                                      batch_size=params['batch_size']))
    run_tests('dlbs_tensorrt_caffe_models_test', dict(type='dlbs', backend='tensorrt', models='dlbs_caffe'),
              models_dir, get_tensorrt_exec(), params)
    shutil.rmtree(models_dir)


def test_models() -> None:
    """ Test onnx (or other - prototxt) models.
    One possible source of models is the ONNX models repository here: https://github.com/onnx/models
    """
    params = dict(batch_size=1, gpus='0', dtype='float32', num_warmup_batches=1, num_batches=1,
                  tensorrt_exec=get_tensorrt_exec())
    models_dir = os.environ.get(
        'DLBS_TEST_MODELS',
        os.path.expandvars('${HOME}/projects/onnx_models/vision/classification/resnet/model')
    )
    run_tests('dlbs_tensorrt_user_test', dict(type='dlbs', backend='tensorrt', models='user'),
              models_dir, get_tensorrt_exec(), params)


def run_tests(run_name: str, tags: t.Optional[t.Mapping],
              models_dir: str, tensorrt_exec: t.Optional[str],
              params: t.Optional[t.Mapping]) -> None:
    """ Main test function.
    Inputs: Models directory (artifact), Path to TensorRT file (artifact), benchmark parameters.
    Outputs: An artifact of type 'DLBSBenchmarkResult'.
    """
    with mlflow.start_run(run_name=run_name, tags=tags):
        tags = dict(dict(type='dlbs', backend='tensorrt'), **(tags or {}))
        tensorrt_exec = tensorrt_exec or get_tensorrt_exec()
        params = dict(dict(batch_size=1, gpus='0', dtype='float32', num_warmup_batches=1, num_batches=1),
                      **(params or {}))

        mlflow.log_param('models', models_dir)
        mlflow.log_params(params)

        logs_dir: str = os.path.join(get_mlflow_artifact_uri(), 'logs')

        models = CafeModels.get_models(models_dir)
        mlflow.log_metric('num_benchmarks', len(models))
        os.makedirs(logs_dir, exist_ok=True)
        logger.info("Running %d benchmarks with log files in %s", len(models), logs_dir)
        passed_benchmarks, failed_benchmarks = 0, 0
        mlflow.log_metrics(dict(passed_benchmarks=passed_benchmarks, failed_benchmarks=failed_benchmarks), step=0)
        for idx, model in enumerate(models):
            # Run benchmark
            log_file = os.path.join(logs_dir, f'{idx + 1}.log')
            logger.info("Running %04d/%04d benchmark with log = %s", idx + 1, len(models), log_file)
            with open(log_file, 'wt') as output_f:
                cmd: t.List[str] = [
                    tensorrt_exec, f"--gpus={params['gpus']}", f"--model={model['title']}",
                    f"--model_file={model['model_path']}", f"--batch_size={params['batch_size']}",
                    f"--dtype={params['dtype']}", f"--num_warmup_batches={params['num_warmup_batches']}",
                    f"--num_batches={params['num_batches']}"
                ]
                process = subprocess.Popen(cmd, stdout=output_f, stderr=output_f)
                process.wait()
            # Get benchmark status and log progress
            if get_benchmark_status(log_file) == BenchmarkStatus.SUCCESS:
                passed_benchmarks += 1
            else:
                failed_benchmarks += 1
            mlflow.log_metrics(dict(passed_benchmarks=passed_benchmarks, failed_benchmarks=failed_benchmarks),
                               step=idx + 1)
        logger.info("Done with passed_benchmarks = %d, failed_benchmarks = %d", passed_benchmarks, failed_benchmarks)


def main():
    """ Run inference tests with TensorRT DLBS benchmark backend.

    Useful environment variables:
        MLFLOW_TRACKING_URI, DLBS_TENSORRT_EXEC, DLBS_TEST_MODELS
    Example usage:
        $ export MLFLOW_TRACKING_URI=${HOME}/.mlflow
        $ export DLBS_TENSORRT_EXEC=${HOME}/projects/dlbs/build/tensorrt
        $ DLBS_TEST_MODELS= python ./tests/test_models.py dlbs
        $ DLBS_TEST_MODELS=${HOME}/projects/onnx_models/vision/classification/resnet/model python ./tests/test_models.py
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    if not os.environ.get('DLBS_TEST_MODELS', None):
        test_dlbs_caffe_models()
    else:
        test_models()


if __name__ == '__main__':
    main()
