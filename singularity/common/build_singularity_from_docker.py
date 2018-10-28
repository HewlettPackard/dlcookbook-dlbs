#!/lvol/sfleisch/anaconda3/bin/python
from __future__ import print_function
import os, sys, re, argparse, subprocess, shlex, tempfile
"""
See: 	https://github.com/TACC/docker2singularity
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', '-t', type=str, required=False, default=None,\
                        help="""$oathtoken for nvcr.io. Get it from ngc.nvidia.com.
                                It isn't necessary to use if docker login has already been executed.
                             """)
    parser.add_argument('--docker_image', '-d', type=str, required=True, default=None,
                        help="""The Docker image to convert.  Use the path in the repository.
                                E.g., nvcr.io/nvidia/tensorflow:18.08-py3
                             """)
    # change the default to your pogram location
    parser.add_argument('--build_command','-b',type=str,default='/usr/bin/sudo /opt/singularity/bin/singularity build',
                        help='The singularity build command - requires sudo.')

    args = parser.parse_args()
    print("token: ",args.token)

    if args.docker_image is None:
        parser.print_usage()
        raise ArgumentError("--docker_image | -d is a required switch.")
    if args.token is not None:
        ret=subprocess.check_output(shlex.split('docker login -u="$oauthtoken" -p={} nvcr.io'.format(args.token)))
    print("Pulling image, {}, from the hub.".format(args.docker_image))
    ret=subprocess.check_output(shlex.split("docker pull {}".format(args.docker_image)))
    print('ret: ',ret)
    print("Attempting to build the Singularity image.")
    popen=subprocess.Popen(shlex.split("docker run -v /var/run/docker.sock:/var/run/docker.sock "+\
                                   "-v {}:/output --privileged -t ".format(os.getcwd())+\
                                   "--rm tacc/docker2singularity {}".format(args.docker_image)),
                                   universal_newlines=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    simage=None
    simage_new=None
    for i,stdout_line in enumerate(iter(popen.stdout.readline,"")):
        stdout_line=stdout_line.strip()
        print(i,':',stdout_line)
        if simage==None:
            m=re.search('Done. Image can be found at: (.*img)',stdout_line)
            if m:
                simage=os.path.basename(m.group(1))
                m1=re.match('(.*?)-20.*?\.img',simage)
                if m1:
                    simage_new=m1.group(1)+'.img'
                else:
                    simage=None
    print('>>Singularity image was created.<<')
    output=popen.communicate()[0]
    exitcode=popen.returncode
    print(simage)
    if simage is not None and simage_new is not None:
        os.rename(simage,simage_new)
        print('renamed {} -> {}'.format(simage,simage_new))

    build_command=args.build_command
    simage=simage_new
    with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as recipe:
        recipe.write("Bootstrap: localimage\n")
        recipe.write("From:{}\n".format(simage))
        recipe.write("%post\n")
        recipe.write("    mkdir -p /usr/local/nvidia\n")
        recipe.write("    ln -s /.singularity.d/libs /usr/local/nvidia/lib64\n")
        recipe.write("%runscript\n")
        recipe.write("    echo \"This Singularity image converted from {}\"\n".format(simage))
        recipe.flush()
    

        cmd=shlex.split("{} {} {}".format(build_command, simage,recipe.name))
        print("command: {}".format(cmd))
        popen=subprocess.Popen(cmd,universal_newlines=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for i,stderr_line in enumerate(iter(popen.stderr.readline,"")):
            stderr_line=stderr_line.strip()
            print(i,':',stderr_line)
        for i,stdout_line in enumerate(iter(popen.stdout.readline,"")):
            stdout_line=stdout_line.strip()
            print(i,':',stdout_line)
        print('>>done<<')
        output=popen.communicate()[0]
        exitcode=popen.returncode

if __name__=="__main__":
    main()
