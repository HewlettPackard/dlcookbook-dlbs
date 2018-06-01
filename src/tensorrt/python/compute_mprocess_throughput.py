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

import sys
import json

if len(sys.argv) == 1:
    print("Usage: %s FNAME [N]" % sys.argv[0])
    exit(1)

# Input file name with benchmark results
fname=sys.argv[1]
# Remove this number of items from start/end.
N=50
if len(sys.argv) >= 3:
    N = int(sys.argv[2])

with open(fname, 'r') as fobj:
    benchmarks = json.load(fobj)

benchmarks = benchmarks['data']
# The 'benchmarks' is a dictionary with single key 'data'. It is an array of results
# of individual benchmarks. Each array is a dictionary containing the following
# keys:
#    exp.replica_batch     Per-GPU batch size
#    results.throughput    Throughput computed by a benchmarking suite
#    results.time_data     Duration in milliseconds of individual batches.

adjusted_mean_time = 0
ngpus = 0
for (i,benchmark) in enumerate(benchmarks):
    times = benchmark['results.time_data'][N:-N]
    adjusted_mean_time += sum(times) / len(times)
    ngpus += benchmark['exp.num_gpus']
    print("Benchmark %d (per GPU perf): own throughput %f, effective throughput %f" % (i, benchmark['results.throughput'], benchmark['results.mgpu_effective_throughput']))

adjusted_mean_time = adjusted_mean_time /  len(benchmarks)
adjusted_throughput = ngpus * 1000.0 * benchmark['exp.replica_batch'] / adjusted_mean_time
print("Adjusted throughput: %f" % adjusted_throughput)
