from matplotlib import pyplot as plt
import os
import sys
import subprocess
import numpy as np

os.environ['TVM_LOG_DEBUG']='0'
plt.style.use('ggplot')

def run_benchmark(path, str_list):
    result = [0.0] * len(str_list)
    try:
        proc = subprocess.Popen(path, shell=True, stdout=subprocess.PIPE)
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            for i, s in enumerate(str_list):
                if line.startswith(bytes(s, encoding = "utf8")):
                    result[i] = float(line.decode().split(':')[-1])
    except:
        return [0.0] * len(str_list)
    # print(path, str_list, result)
    return result

def set_num_threads(num):
    os.environ['TVM_NUM_THREADS']=str(num)
    
    os.environ['BLIS_NUM_THREADS']=str(num)
    # os.environ['OMP_PLACES']="cores"
    # os.environ['OMP_PROC_BIND']="close"

    os.environ['MKL_NUM_THREADS']=str(num)
    os.environ['MKL_DYNAMIC']="FALSE"
    os.environ['KMP_AFFINITY']="granularity=fine,compact,1,0"

USAGE = """
    python gemm_perf.py float32
    or 
    python gemm_perf.py float64
"""
def main(argv):
    if (len(argv) != 2) or (argv[1] != 'float32' and argv[1] != 'float64'):
        print(USAGE)
        sys.exit(255)
    
    dtype = argv[1]

    threads = []
    impls = ['numpy', 'tvm_without_tune', 'tvm_autotvm', 'tvm_autoscheduler', 'blis']
    speeds = {'numpy':[], 'tvm_without_tune':[], 'tvm_autotvm':[], 'tvm_autoscheduler':[], 'blis':[]}
    exec = {'numpy':(f'python tvm_without_tune.py {dtype}', 'Numpy'),
        'tvm_without_tune':(f'python tvm_without_tune.py {dtype}', 'TVM'),
        'tvm_autotvm':(f'python tvm_autotvm_tune.py {dtype}', 'TVM'),
        'tvm_autoscheduler':(f'python tvm_autoscheduler_tune.py {dtype}', 'TVM'),
        'blis':(f'./blis_{dtype}.x', 'BLIS')
        }
    
    for num_threads in [1, 4, 8, 12, 16]:
        threads.append(num_threads)
        set_num_threads(num_threads)
        for impl in impls:
            speeds[impl].append(
                2.0 * 1024 * 1024 * 1024 / run_benchmark(exec[impl][0], [exec[impl][1]])[0] / 1000000000.0
                )

    with open(f'results_{dtype}.txt', 'w') as rf:
        rf.write("threads:")
        rf.write(str(threads))
        rf.write('\n')
        rf.write("speeds")
        rf.write(str(speeds))

    x = np.arange(len(threads))
    width = 0.1
    plt.bar(x - 2.0*width, speeds['numpy'], width, label='numpy')
    plt.bar(x - width, speeds['tvm_without_tune'], width, label='tvm_without_tune')
    plt.bar(x , speeds['tvm_autotvm'], width, label='tvm_autotvm')
    plt.bar(x + width, speeds['tvm_autoscheduler'], width, label='tvm_autoscheduler')
    plt.bar(x + 2.0 * width, speeds['blis'], width, label='blis')
    plt.ylabel('GFlops')
    plt.xlabel('Number of Threads')
    plt.title(f'1024x1024x1024 {dtype} gemm perf test on Numpy(MKL), TVM and BLIS')
    plt.xticks(x, labels=threads)
    plt.legend()
    
    plt.savefig(f'results_{dtype}.png', dpi=400, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main(sys.argv)








