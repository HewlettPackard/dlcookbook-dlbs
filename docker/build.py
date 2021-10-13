import os
import sys
import json
import shutil
import logging
import argparse
import typing as t
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


def dlbs_log(labels: t.Optional[t.Union[t.Text, t.List[t.Text]]] = None, level: int = logging.INFO, **kwargs) -> None:
    labels = labels or []
    if isinstance(labels, str):
        labels = [labels]
    record: t.Dict = dict(labels=labels, time=datetime.utcnow().isoformat(), record=kwargs)
    logger.log(level, ":::DLBS_LOG %s", json.dumps(record))


def execute(cmd: t.Union[t.Text, t.Iterable[t.Text]], die_on_error: bool = True) -> int:
    """ Run command using `os.system` call.
    Args:
        cmd: Command to execute.
        die_on_error: If true, raise an exception.
    Returns:
        Execution code returned by `os.system`.
    """
    if isinstance(cmd, list):
        cmd = ' '.join(cmd)
    dlbs_log('execute', cmd=cmd)
    exec_code: int = os.system(cmd)
    if exec_code != 0 and die_on_error:
        raise RuntimeError(f"execute(cmd={cmd}, exec_code={exec_code})")
    return exec_code


class DockerImage(object):
    """ Class to store parameters of a docker image - prefix (user), name (repo) and tag """
    class Prefix:
        """ This is really a user name that's part of the docker image: PREFIX/NAME:VERSION. """
        HPE: t.Text = 'hpe'     # Legacy prefix, do not use it.
        DLBS: t.Text = 'dlbs'   # This should be used (DLBS - Deep Learning Benchmark Suite).

    def __init__(self, prefix: t.Optional[t.Text], name: t.Text, tag: t.Text) -> None:
        """
        Args:
            prefix: Image prefix (user name).
            name: Image name (repository).
            tag: Image tag.
        Full name is PREFIX/NAME:VERSION
        """
        self.prefix = prefix or DockerImage.Prefix.DLBS
        self.name = name
        self.tag = tag

    @property
    def full_name(self) -> t.Text:
        return f"{self.prefix}/{self.name}:{self.tag}"

    @property
    def relative_path(self) -> t.Text:
        return os.path.join(self.name, self.tag)


class BuildHook(object):
    """ Base class for all frameworks that require hooks (pre/post build steps). """
    def __init__(self, build_context: t.Text, image: DockerImage, version: t.Optional[t.Text]) -> None:
        """
        Args:
            build_context: Directory that is docker build context.
            image: Parameters of the image to build.
            version: Some kind of identifier of a framework to build/install inside docker image (e.g., GitHub commit).
        """
        self.build_context = build_context
        self.image = image
        self.version = version

    def pre_build(self) -> None:
        """ Run this before building an image. """
        ...

    def post_build(self) -> None:
        """ Run this after an image has been built. """
        ...


class TensorRTHook(BuildHook):
    """
    Old versions of docker files used external TensorRT packages and base CUDA images because at that time NGC did not
    exist or did not provide TensorRT images. Starting December 2018, DLBS can use TensorRT images from NGC.
    '18.12', '21.08'
    """
    def pre_build(self) -> None:
        deb_file: t.Optional[t.Text] = self.version
        if deb_file and deb_file.endswith('.deb'):
            if not os.path.isfile(os.path.join(self.build_context, deb_file)):
                raise ValueError(
                    f"Will not build TensorRT ({self.build_context}) because TensorRT DEB file "
                    f"({self.build_context}/{deb_file}) not found. You must copy corresponding package into that "
                    "folder. You can get it from NVIDIA developer site."
                )
        destination: t.Text = os.path.join(self.build_context, 'tensorrt')
        if os.path.exists(destination):
            shutil.rmtree(destination)
        source = os.path.abspath(os.path.join(self.build_context, '..', '..', '..', 'src', 'tensorrt'))
        if not os.path.isdir(source):
            raise IOError(f"TensorRT benchmark does not exist ({source}).")
        execute(['cp', '-r', source, self.build_context])

    def post_build(self) -> None:
        shutil.rmtree(os.path.join(self.build_context, 'tensorrt'))


class OpenVinoHook(BuildHook):
    def pre_build(self) -> None:
        if self.image.tag == '19.09-custom-mkldnn':
            return
        # Once base image with OpenVINO becomes available, this will not be required anymore.
        # http://registrationcenter-download.intel.com/akdlm/irc_nas/15792/l_openvino_toolkit_p_2019.2.275.tgz
        # http://registrationcenter-download.intel.com/akdlm/irc_nas/15944/l_openvino_toolkit_p_2019.3.334.tgz
        sources: t.Text = os.path.join(self.build_context, self.version[6:])
        archive: t.Text = sources + '.tgz'
        if not os.path.exists(sources):
            if not os.path.exists(archive):
                url = f"http://registrationcenter-download.intel.com/akdlm/irc_nas/{self.version}.tgz"
                execute(f'cd {self.build_context} && wget {url};')
            execute(f'cd {self.build_context} && tar -xf l_openvino_toolkit*;')

    def post_build(self) -> None:
        ...


class ImageBuilder(object):

    @staticmethod
    def list_images() -> None:
        """ List all benchmark images that need to be built before running benchmarks.
        These are the images defined in ./docker directory, e.g. tensorrt/21.08.
        """
        root_dir: t.Text = os.path.dirname(__file__)
        paths: t.Iterable[t.Tuple] = (
            path.relative_to(root_dir).parts[:-1] for path in Path(root_dir).rglob('Dockerfile')
        )
        benchmarks: t.Dict = defaultdict(list)
        for path in paths:
            if len(path) == 2:
                benchmarks[path[0]].append(path[1])
        print("Benchmark Images:")
        for benchmark, versions in benchmarks.items():
            print(f"\t{benchmark}: {sorted(versions)}")

    @staticmethod
    def parse_image_path(image_path: t.Text, prefix: t.Optional[t.Text] = None) -> DockerImage:
        """ Split benchmark image into name and tag, verify image path points to a valid directory.
        Args:
            image_path: Image path provided by a user, e.g., tensorrt/21.08.
            prefix: Image prefix to use.
        Returns:
            A structure that provides docker image information (prefix, name and tag).
        """
        parts = image_path.split(os.path.sep)
        if len(parts) == 3 and (parts[2] == 'Dockerfile' or parts[2] == ''):
            del parts[2]
        if len(parts) != 2:
            raise ValueError(f"Invalid image path ({image_path}). Expecting FRAMEWORK/VERSION.")
        full_path = os.path.join(os.path.dirname(__file__), image_path)
        if not os.path.isdir(full_path):
            raise ValueError(f"Invalid image path ({image_path}). Not a directory: {full_path}.")
        if not os.path.isfile(os.path.join(full_path, 'Dockerfile')):
            raise ValueError(f"Invalid image path ({image_path}). Not a file: {os.path.join(full_path, 'Dockerfile')}.")
        return DockerImage(prefix, parts[0], parts[1])

    def __init__(self, images: t.Optional[t.Union[t.Text, t.List[t.Text]]] = None, prefix: t.Optional[t.Text] = None,
                 build_args: t.Optional[t.Dict] = None, run_args: t.Optional[t.Dict] = None) -> None:
        """
        Args:
            images: List of images to build.
            prefix: If present, use this for all images.
            build_args: Docker build arguments. The following are supported:
                - `version`: Some kind of identifier of the framework to build inside a docker image. It's optional,
                  and there's a good chance it's going to be deprecated. Example values: GitHub commit, DEB package
                  name, etc. Its value is image-specific.
            run_args: Docker runtime arguments.
                - `docker`: Docker executable (docker, nvidia-docker, sudo docker, podman, etc.).
        """
        if isinstance(images, str):
            images = [images]
        self.images = images or []
        self.prefix = prefix or DockerImage.Prefix.DLBS
        self.build_args = build_args or dict()
        self.run_args = run_args or dict()

    @staticmethod
    def load_versions(file_name: t.Optional[t.Text] = None) -> t.Dict:
        """ Load framework versions from a file. It's highly likely, this functionality is going to be deprecated.
        Args:
            file_name: Path to `versions` file.
        Returns:
            Dictionary that maps frameworks names and tags to framework versions.
        """
        if not file_name:
            file_name = os.path.join(os.path.dirname(__file__), 'versions.json')
        if not os.path.isfile(file_name):
            raise IOError(f"Versions file ({file_name}) does not exist.")
        with open(file_name, 'r') as fh:
            versions = json.load(fh)
        return versions

    def get_version(self, versions: t.Dict, image: DockerImage) -> t.Optional[t.Text]:
        """ Return version of the framework to use in a docker image.
        In legacy or custom benchmark images, we build DL or ML frameworks from sources. In this case, we need to know
        a commit or some other string that identifies how to retrieve the right framework version. This seems to be not
        very good decision choice, but this is what we have for now.
        Moving forward (what's the case with NGC images for instance), a new docker file needs to be present for each
        new framework version.
        Args:
            versions: Dictionary of versions loaded from a file.
            image: Parameters of the docker image to build.
        Returns:
            Framework version for this docker image.
        """
        version = self.build_args.get('version', None)
        if version is None:
            for tag in (image.tag, 'default'):
                version = versions.get(image.name, dict()).get(tag, None)
                if version is not None:
                    dlbs_log('version', image=f'{image.prefix}/{image.name}:{tag}', version=version, source='file')
                    break
        else:
            dlbs_log('version', image=image.full_name, version=version, source='CLI')
        dlbs_log('version', image=image.full_name, version=version)
        return version

    @staticmethod
    def get_build_args(version: t.Optional[t.Text]) -> t.Text:
        """ Return docker build arguments. They include version and HTTP(S) variables.
        Args:
            version: Framework version.
        Returns:
            String containing formatted docker build arguments.
        """
        args = dict(
            version=version,
            http_proxy=os.environ.get('http_proxy', None),
            https_proxy=os.environ.get('https_proxy', None)
        )
        return ' '.join([f"--build-arg {k}={v}" for k, v in args.items() if v])

    @staticmethod
    def get_framework_hook(build_context: t.Text, image: DockerImage, version: t.Text) -> BuildHook:
        """ Return a hook instance for this docker image.
        Args:
            build_context: Docker build context.
            image: Parameters of the docker image to build.
            version: Framework version inside docker image.
        Returns:
            An instance of one of the build hooks.
        """
        hooks: t.Dict = dict(tensorrt=TensorRTHook, openvino=OpenVinoHook)
        return hooks.get(image.name, BuildHook)(build_context, image, version)

    def build(self) -> None:
        """ Build docker image. """
        images: t.List[DockerImage] = [ImageBuilder.parse_image_path(image, self.prefix) for image in self.images]
        versions: t.Dict = ImageBuilder.load_versions()

        dlbs_log('build_parameters', images=self.images, prefix=self.prefix, build_args=self.build_args,
                 run_args=self.run_args)

        exec_summary = []
        for image in images:
            docker: t.Text = self.run_args.get('docker', None) or 'docker'
            version: t.Text = self.get_version(versions, image)
            build_args: t.Text = ImageBuilder.get_build_args(version)
            build_context: t.Text = str((Path(__file__).parent / image.name / image.tag).absolute())

            hook = ImageBuilder.get_framework_hook(build_context, image, version)
            hook.pre_build()

            status: int = execute([docker, 'build', '-t', image.full_name, build_args, build_context])
            exec_summary.append(dict(image=image.full_name, docker=docker, build_args=build_args, status=status,
                                     build_context=build_context))

            hook.post_build()

        for exec_record in exec_summary:
            dlbs_log('build_summary', **exec_record)


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--docker', type=str, required=False, default='docker',
        help="An executable for docker. Most common value is 'docker', but also can be 'nvidia-docker'. In certain "
             "cases, when current user does not belong to a 'docker' group, it should be 'sudo docker'. Default value "
             "is 'docker'."
    )
    parser.add_argument(
        '--prefix', type=str, required=False, default='dlbs',
        help="Set image prefix 'prefix/..'. Default is 'hpe' or 'dlbs'. By default, all images have the following "
             "name: 'prefix/framework:tag' where framework is a folder in this directory and tag is a sub-folder in "
             "framework's folder. If prefix is empty, no prefix will be used and image name will be set to "
             "'framework:tag'. Default values for docker images in benchmarking suite assume the prefix exists "
             "(dlbs/). If you want to use different prefix, make sure to override image name when running experimenter."
    )
    parser.add_argument(
        '--version', type=str, required=False, default=None,
        help="If supported by a docker file, framework COMMIT to clone from github. Default value is taken from "
             "'versions' file located in this directory. This is not a specific version like 1.4.0, rather it is a "
             "commit tag. All docker files will execute the 'git reset --hard \$version' to use particular project "
             "state. Default is set to 'master' in docker files. If user provides this command line argument, this "
             "commit will be used for ALL builds (user can provide more than one image to build). So, this may be "
             "useful when building one docker image or building docker images for one particular framework."
    )
    parser.add_argument(
        'images', nargs='?', help="List of benchmark images to build. They are identified by relative paths in this "
                                  "directory, for instance, 'tensorrt/21.08'. For the list of available images, run "
                                  "this script without parameters, e.g., 'python build.py'."
    )
    return parser.parse_args()


def main():
    if len(sys.argv) == 1:
        ImageBuilder.list_images()
        return

    args: argparse.Namespace = parse_cli_arguments()
    builder = ImageBuilder(args.images, args.prefix, build_args=dict(version=args.version),
                           run_args=dict(docker=args.docker))
    builder.build()


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    main()
