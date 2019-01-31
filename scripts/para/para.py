from __future__ import print_function

from pssh.clients import ParallelSSHClient

hosts = ['myhost1', 'myhost2']
client = ParallelSSHClient(hosts)

output = client.run_command('uname')
for host, host_output in output.items():
    for line in host_output.stdout:
        print(line)
