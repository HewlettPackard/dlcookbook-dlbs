# (c) Copyright [2017] Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple http server that provides basic capabilities on monitoring
progress of a benchmarking experiment.

$ python simple_server.py PROGRESS_FILE PORT

* ``PROGRESS_FILE`` A path to JSON file that's updated by an experimenter
   process. It's the very same file that you pass to experimenter script with
   --progress-file command line argument.
* ``PORT`` A port number to use.

On running instance of simple server can server one experiment session.
If due to some reason you want to run multiple experiments at the same time
on same machine, use different progress files and ports:

$ nohup python simple_server.py /dev/shm/experiment1.json 8000 &
$ nohup python simple_server.py /dev/shm/experiment2.json 8001 &

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import json
from dlbs.utils import Six
# Do not want to make `six` module a dependency for the project.
if Six.PY3:
    from http.server import BaseHTTPRequestHandler, HTTPServer
else:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer


class DLBSHandler(BaseHTTPRequestHandler):
    """HTTP request handler for simple server

    This handler responds to incoming requests in three different ways:
    * ``/`` path -> send back index.html page.
    * ``/api/progress`` path -> send back contents of JSON file.
    * Everything else is forbidden and 403 code is returned.
    """

    INDEX_HTML_FILE = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../../web/index.html'
        )
    )
    PROGRESS_FILE = None

    def _set_headers(self):
        """Depending on request path, sets response header info."""
        if self.path == '/':
            response = 200
            header = ('Content-type', 'text/html')
        elif self.path == '/api/progress':
            response = 200
            header = ('Content-type', 'application/json')
        else:
            response = 403
            header = ('Content-type', 'text/html')
        self.send_response(response)
        self.send_header(header[0], header[1])
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        """Serve GET request."""
        if self.path == '/':
            try:
                with open(DLBSHandler.INDEX_HTML_FILE, 'r') as file_obj:
                    response_data = file_obj.read()
            except IOError as err:
                response_data = "<h1>Internal server error</h1><hr/>" + str(err)
        elif self.path == '/api/progress':
            try:
                with open(DLBSHandler.PROGRESS_FILE, "r") as file_obj:
                    response_data = json.dumps({
                        "status": 0,
                        "status_message": "",
                        "progress": json.load(file_obj)
                    })
            except IOError as err:
                response_data = json.dumps({
                    "status": 1,
                    "status_message": str(err),
                    "progress": {}
                })
        else:
            response_data = None
        self._set_headers()
        if response_data is not None:
            self.wfile.write(response_data)


class DLBSServer(HTTPServer):
    """An http server that accepts application arguments."""
    def run_forever(self, poll_interval=0.5, progress_file=None):
        """Almost synonym for HTTPServer::serve_forever
        Handles requests until an explicit shutdown() request. Poll for
        shutdown every poll_interval seconds

        :param float poll_interval: Time in seconds to poll for shutdown.
        :param str progress_file: A JSON file with current status of a
                                  benchmarking experiment.
        """
        self.RequestHandlerClass.PROGRESS_FILE = progress_file
        HTTPServer.serve_forever(self, poll_interval)


def main():
    """Simple web server entry point."""
    if len(sys.argv) < 3:
        print("Usage: python %s PROGRESS_FILE PORT" % sys.argv[0])
        exit(1)
    server_address = ('', int(sys.argv[2]))
    httpd = DLBSServer(server_address, DLBSHandler)
    httpd.run_forever(progress_file=sys.argv[1])


if __name__ == '__main__':
    main()
