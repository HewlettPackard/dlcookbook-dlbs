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
from __future__ import print_function
import os
import sys
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

class DLBSHandler(BaseHTTPRequestHandler):

    PROGRESS_FILE = None

    def _set_headers(self):
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
        self._set_headers()
        if self.path == '/':
            html_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '../../../web/index.html'
                )
            )
            with open(html_file, "r") as f:
                self.wfile.write(f.read())
        elif self.path == '/api/progress':
            with open(DLBSHandler.PROGRESS_FILE, "r") as f:
                self.wfile.write(f.read())


class DLBSServer(HTTPServer):
    def run_forever(self, poll_interval=0.5, progress_file=None):
        self.RequestHandlerClass.PROGRESS_FILE = progress_file
        HTTPServer.serve_forever(self, poll_interval)


def main():
    if len(sys.argv) < 3:
        print("Usage: python %s PROGRESS_FILE PORT" % sys.argv[0])
        exit(1)
    server_address = ('', int(sys.argv[2]))
    httpd = DLBSServer(server_address, DLBSHandler)
    httpd.run_forever(progress_file=sys.argv[1])


if __name__ == '__main__':
    main()
