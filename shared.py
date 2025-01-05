from typing import List


def gcd(val1: int, val2: int) -> int:
    if val1 == 0 and val2 == 0:
        return -1
    if val1 == 0:
        return val2
    if val2 == 0:
        return val1
    if val1 == val2:
        return val1
    if val1 > val2:
        return gcd(val2, val1 % val2)
    else:
        return gcd(val1, val2 % val1)


def lcm(val1: int, val2: int) -> int:
    return int(abs(val1) * (abs(val2) / gcd(val1, val2)))


def all_occurences_in_string(s, c):
    return [i for i in range(len(s)) if s[i] == c]


def read_file(fn):
    f = open(fn, "r")
    s = f.read()
    f.close()
    return s


def read_file_as_lines(fn) -> List[str]:
    lines = []
    with open(fn, "r") as file:
        for line in file:
            lines.append(line.strip())
    return lines


def read_file_as_single_line(fn):
    s = read_file(fn)
    all_newlines = all_occurences_in_string(s, "\n")
    if len(all_newlines) == 0:
        line_length = len(s)
    else:
        line_length = all_newlines[0]
    num_lines = len(all_newlines) + 1
    s = s.replace("\n", "")
    return s, num_lines, line_length
