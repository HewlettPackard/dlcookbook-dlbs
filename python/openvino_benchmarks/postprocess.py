import sys


def error(message):
    print(message)
    exit(1)


def main():
    fname = sys.argv[1]
    effective_batch = int(sys.argv[2])
    status = 'failure'
    #
    with open(fname) as logfile:
        for line in logfile:
            line = line.strip()
            if line.startswith('Throughput:') and line.endswith('FPS'):
                try:
                    throughput = float(line[11:-3].strip())
                    batch_time = 1000.0 / (throughput / effective_batch)
                    print("__results.throughput__=%f" % throughput)
                    print("__results.time__=%f" % batch_time)
                    status = 'success'
                except Exception as e:
                    print("ERROR parsing log file")
                    print(e)
                    pass
                break
    #
    print("__exp.status__=\"{}\"".format(status))


if __name__ == '__main__':
    main()
