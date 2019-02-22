#!/usr/bin/env python
# encoding: utf-8
"""
@Author     :xinbei
@Software   :Pycharm
@File       :a_start.py
@Time       :2019/2/22 12:13
@Version    :
@Purpose    :
"""
import functools, time


# 计算函数运行时间的装饰器
def run_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        t = time.time()
        result = func(*args, **kw)
        print('%s execute %.6f s' % (func.__name__, time.time() - t))
        return result

    return wrapper


if __name__ == '__main__':
    pass