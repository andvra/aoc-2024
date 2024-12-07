from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Point2f:
    x: float = 0
    y: float = 0

    def __add__(a, b):
        return Point2f(a.x + b.x, a.y + b.y)

    def __sub__(a, b):
        return Point2f(a.x - b.x, a.y - b.y)

    def __truediv__(v, d):
        v.x /= d
        v.y /= d
        return v

    @staticmethod
    def length(v):
        return math.sqrt(v.x * v.x + v.y * v.y)

    @staticmethod
    def normalize(v):
        return v / Point2f.length(v)

    @staticmethod
    def collides(pos, dir, target):
        v = target - pos
        n = Point2f.normalize(v)
        return Point2f.length(n - dir) < 0.001


@dataclass
class Point2i:
    x: int = 0
    y: int = 0

    def __add__(a, b):
        return Point2i(a.x + b.x, a.y + b.y)

    def __sub__(a, b):
        return Point2i(a.x - b.x, a.y - b.y)

    def __mul__(a, b):
        return Point2i(a.x * b, a.y * b)

    @staticmethod
    def collides(pos, dir, target):
        return Point2f.collides(
            Point2f(pos.x, pos.y), Point2f(dir.x, dir.y), Point2f(target.x, target.y)
        )


def all_occurences_in_string(s, c):
    return [i for i in range(len(s)) if s[i] == c]


def read_file(fn):
    f = open(fn, "r")
    s = f.read()
    f.close()
    return s


def read_file_as_lines(fn):
    lines = []
    with open(fn, "r") as file:
        for line in file:
            lines.append(line.strip())
    return lines


def read_file_as_single_line(fn):
    s = read_file(fn)
    all_newlines = all_occurences_in_string(s, "\n")
    line_length = all_newlines[0]
    num_lines = len(all_newlines) + 1
    s = s.replace("\n", "")
    return s, num_lines, line_length


def day4_part1(fn):
    input, num_lines, line_length = read_file_as_single_line(fn)
    all_positions = all_occurences_in_string(input, "X")
    word_to_match = "XMAS"
    cnt_chars = len(word_to_match)
    num_occurences = 0
    for pos in all_positions:
        offsets = []
        row = pos // line_length
        col = pos % line_length
        can_move_left = False
        can_move_right = False
        can_move_up = False
        can_move_down = False
        if col >= cnt_chars - 1:
            can_move_left = True
            offsets.append(-1)
        if col <= line_length - cnt_chars:
            can_move_right = True
            offsets.append(1)
        if row >= cnt_chars - 1:
            can_move_up = True
            offsets.append(-line_length)
        if row <= num_lines - cnt_chars:
            can_move_down = True
            offsets.append(line_length)
        if can_move_left and can_move_up:
            offsets.append(-line_length - 1)
        if can_move_left and can_move_down:
            offsets.append(line_length - 1)
        if can_move_right and can_move_up:
            offsets.append(-line_length + 1)
        if can_move_right and can_move_down:
            offsets.append(line_length + 1)
        for offset in offsets:
            s = ""
            for idx in range(4):
                s += input[pos + idx * offset]
            if s == word_to_match or s[::-1] == word_to_match:
                num_occurences += 1
    return num_occurences


def day4_part2(fn):
    input, num_lines, line_length = read_file_as_single_line(fn)
    all_positions = all_occurences_in_string(input, "A")
    num_occurences = 0
    for pos in all_positions:
        row = pos // line_length
        col = pos % line_length
        ok_row = row >= 1 and row < num_lines - 1
        ok_col = col >= 1 and col < line_length - 1
        if ok_row and ok_col:
            s1 = input[pos - line_length - 1] + input[pos + line_length + 1]
            s2 = input[pos - line_length + 1] + input[pos + line_length - 1]
            if (s1 == "MS" or s1 == "SM") and (s2 == "MS" or s2 == "SM"):
                num_occurences += 1
    return num_occurences


def day5_common(fn) -> tuple[dict, list[str], list[str]]:
    lines = read_file_as_lines(fn)
    idx_line_split = 0
    for idx_line, line in enumerate(lines):
        if not line:
            idx_line_split = idx_line

    lines_rule = lines[:idx_line_split]
    lines_book = lines[idx_line_split + 1 :]

    rule_lines = [line.split("|") for line in lines_rule]
    rules = {}

    for before, after in rule_lines:
        if before in rules:
            rules[before].append(after)
        else:
            rules[before] = [after]

    books_valid = []
    books_invalid = []

    for book in lines_book:
        correct_order = True
        pages = book.split(",")
        num_pages = len(pages)
        for idx_page in range(num_pages - 1, 0, -1):
            page_after = pages[idx_page]
            has_rules_for_page = page_after in rules
            if not has_rules_for_page:
                continue
            pages_after_current = rules[page_after]
            for idx_check in range(idx_page - 1, -1, -1):
                page_before = pages[idx_check]
                if page_before in pages_after_current:
                    correct_order = False
                    break
        if correct_order:
            books_valid.append(book)
        else:
            books_invalid.append(book)

    return rules, books_valid, books_invalid


def day5_part1(fn):
    _, books_valid, _ = day5_common(fn)

    ret = 0

    for book in books_valid:
        pages = book.split(",")
        num_pages = len(pages)
        ret += int(pages[num_pages // 2])
    return ret


def day5_part2(fn):
    rules, _, books_invalid = day5_common(fn)

    ret = 0

    for book in books_invalid:
        valid_order = []
        pages = book.split(",")
        num_pages = len(pages)
        valid_order.append(pages[0])
        for page in pages[1:]:
            was_added = False
            for idx, cur_added in enumerate(valid_order):
                if (page in rules) and (cur_added in rules[page]):
                    valid_order.insert(idx, page)
                    was_added = True
                    break
            if not was_added:
                valid_order.append(page)
        ret += int(valid_order[num_pages // 2])
    return ret


def day6_part1(fn):
    lines = read_file_as_lines(fn)
    line_width = len(lines[0])
    num_lines = len(lines)
    points = []
    pos_guard = Point2i()
    dir = Point2i()

    for idx_line, line in enumerate(lines):
        positions = all_occurences_in_string(line, "#")
        for pos in positions:
            points.append(Point2i(pos, idx_line))
        for idx_char in range(line_width):
            c = line[idx_char]
            if c in "^>v<":
                pos_guard = Point2i(idx_char, idx_line)
                dir_x = 0
                dir_y = 0
                if c == "<":
                    dir_x = -1
                if c == ">":
                    dir_x = 1
                if c == "^":
                    dir_y = -1
                if c == "v":
                    dir_y = -1
                dir = Point2i(dir_x, dir_y)

    done = False
    visited = [[False] * line_width for _ in range(num_lines)]
    visited[pos_guard.y][pos_guard.x] = True

    while not done:
        pt_collision = None
        distance_collision = 10000
        for point in points:
            does_collide = Point2i.collides(pos_guard, dir, point)
            if does_collide:
                new_distance = Point2f.length(point - pos_guard)
                if new_distance < distance_collision:
                    pt_collision = point
                    distance_collision = new_distance
        if pt_collision:
            pt_before = pt_collision - dir
            num_steps = int(Point2f.length(pt_before - pos_guard))
            for idx_step in range(1, num_steps + 1):
                cur_pos = pos_guard + dir * idx_step
                visited[cur_pos.y][cur_pos.x] = True
            pos_guard = pt_before
            dir = Point2i(-dir.y, dir.x)
        else:
            cur_pos = pos_guard + dir
            while (cur_pos.x >= 0 and cur_pos.x < line_width) and (
                cur_pos.y >= 0 and cur_pos.y < num_lines
            ):
                visited[cur_pos.y][cur_pos.x] = True
                cur_pos += dir

            done = True

    num_visited = sum(x.count(True) for x in visited)

    return num_visited


def aoc_2024():
    for num_day in range(1, 26):
        fn_test = f"input/day{num_day}-test.txt"
        fn_real = f"input/day{num_day}-real.txt"
        for part in range(1, 3):
            func_name = f"day{num_day}_part{part}"
            func = None
            try:
                func = eval(func_name)
            except:
                continue
            res_test = func(fn_test)
            res_real = func(fn_real)
            print(f"Day {num_day} (part {part}): {res_test} / {res_real}")


if __name__ == "__main__":
    aoc_2024()
