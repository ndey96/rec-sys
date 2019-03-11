import begin
import multiprocessing
import time


def worker(varying_data, aux_data):
    t = 0
    for j in range(1, 10000):
        t += varying_data
    return t


aux_data = None


def initializer(init_data):
    global aux_data
    aux_data = init_data


def with_initializer_worker_wrapper(varying_data):
    return worker(varying_data, aux_data)


@begin.subcommand
def with_initializer():
    iterations = 10
    for i in range(3, 8):
        start_time = time.time()
        aux_data = [i] * pow(10, i)
        pool = multiprocessing.Pool(4, initializer, (aux_data,))
        data = [1 for x in range(1, 1001)]
        tmp = 0
        for i in range(1, iterations):
            tmp = sum(pool.map(with_initializer_worker_wrapper, data))
        pool.close()
        pool.join()
        pool.terminate()
        end_time = time.time()
        secs_per_iteration = (end_time - start_time) / iterations
        print("aux_data {0:>10,} ints : {1:>6.6f} secs per iteration {2}"
              .format(len(aux_data), secs_per_iteration, tmp))


def without_initializer_worker_wrapper(data):
    return worker(*data)


@begin.subcommand
def without_initializer():
    iterations = 10
    for i in range(3, 8):
        start_time = time.time()
        aux_data = [i] * pow(10, i)
        pool = multiprocessing.Pool(4)
        data = [(1, aux_data) for x in range(1, 1001)]
        tmp = 0
        for i in range(1, iterations):
            tmp = sum(pool.map(without_initializer_worker_wrapper, data))
        pool.close()
        pool.join()
        pool.terminate()
        end_time = time.time()
        secs_per_iteration = (end_time - start_time) / iterations
        print("aux_data {0:>10,} ints : {1:>6.6f} secs per iteration {2}"
              .format(len(aux_data), secs_per_iteration, tmp))


@begin.start
def entry():
    pass