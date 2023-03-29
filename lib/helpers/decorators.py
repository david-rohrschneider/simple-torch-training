from functools import wraps
from time import time


def stop_time(function):
    @wraps(function)
    def timer_wrapper(*args, **kwargs):
        start_time = time()
        result = function(*args, **kwargs)
        end_time = time()
        execution_time = end_time - start_time
        if result:
            return result, execution_time
        return execution_time

    return timer_wrapper
