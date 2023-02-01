

def run_print(lock, io_path, *str_list):
    lock.acquire()
    try:
        with open(io_path, 'a') as f:
            for one_str in str_list:
                f.write(one_str)
                f.write('\n')
    except Exception as e:
        lock.release()
        print('run_print has an error\n', e)
        return
    lock.release()
