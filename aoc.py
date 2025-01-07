import sys
import time
from day1to5 import *
from day6to10 import *
from day11to15 import *
from day16to20 import *
from day21to25 import *


def aoc_2024(run_real=True, single_day=None):
    day_start = 1
    day_end_excl = 26
    if single_day != None:
        day_start = single_day
        day_end_excl = single_day + 1
    t_start = time.time()
    for num_day in range(day_start, day_end_excl):
        fn_test = f"input/day{num_day}-test.txt"
        fn_real = f"input/day{num_day}-real.txt"
        for part in [1, 2]:
            func_name = f"day{num_day}_part{part}"
            func = None
            try:
                func = eval(func_name)
            except:
                continue
            res_test = func(fn_test)
            res_real = 0
            if run_real:
                res_real = func(fn_real)
            print(f"Day {num_day} (part {part}): {res_test} / {res_real}")
    t_end = time.time()
    print("Elapsed time (s):", int(10000 * (t_end - t_start)) / 10000)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        single_day = int(sys.argv[1])
        aoc_2024(single_day=single_day)
    else:
        aoc_2024()
