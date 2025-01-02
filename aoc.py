from dataclasses import dataclass
import math
import re
import sys
import itertools
import numpy as np
from typing import List


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


def day1_part1(fn):
    lines = read_file_as_lines(fn)
    list1 = []
    list2 = []
    for line in lines:
        a, b = map(int, line.split())
        list1.append(a)
        list2.append(b)
    list.sort(list1)
    list.sort(list2)
    tot_diff = 0
    for a, b in zip(list1, list2):
        tot_diff = tot_diff + abs(a - b)
    return tot_diff


def day1_part2(fn):
    lines = read_file_as_lines(fn)
    list1 = []
    list2 = []
    for line in lines:
        a, b = map(int, line.split())
        list1.append(a)
        list2.append(b)
    num_map = {}
    for el in list2:
        if el in num_map:
            num_map[el] = num_map[el] + 1
        else:
            num_map[el] = 1

    score = 0
    for el in list1:
        if el in num_map:
            score = score + el * num_map[el]
    return score


def day2_part1(fn):
    lines = read_file_as_lines(fn)

    num_safe = 0
    for line in lines:
        numbers = list(map(int, line.split()))
        if numbers[0] > numbers[1]:
            numbers = numbers[::-1]
        safe = True
        for idx_number in range(len(numbers) - 1):
            cur = numbers[idx_number]
            next = numbers[idx_number + 1]
            if next <= cur or next - cur > 3:
                safe = False
        if safe:
            num_safe = num_safe + 1
    return num_safe


def day2_first_error(numbers):
    num_numbers = len(numbers)
    for idx_number in range(num_numbers - 1):
        cur = numbers[idx_number]
        next = numbers[idx_number + 1]
        diff = next - cur
        is_invalid = (diff < 1) or (diff > 3)
        if is_invalid:
            return idx_number
    return None


def day2_part2(fn):
    lines = read_file_as_lines(fn)

    num_safe = 0
    for line in lines:
        numbers = list(map(int, line.split()))
        safe = False
        for idx_dir in [0, 1]:
            if idx_dir == 0:
                cur_numbers = numbers[:]
            if idx_dir == 1:
                cur_numbers = numbers[::-1]
            first_error = day2_first_error(cur_numbers)
            cur_safe = True
            if first_error != None:
                arr1 = cur_numbers[:first_error] + cur_numbers[first_error + 1 :]
                arr2 = cur_numbers[: first_error + 1] + cur_numbers[first_error + 2 :]
                cur_safe = day2_first_error(arr1) == None
                cur_safe = cur_safe or day2_first_error(arr2) == None
            safe = safe or cur_safe
        if safe:
            num_safe = num_safe + 1
    return num_safe


def day3_result_from_line(line):
    start_indices = [m.start() for m in re.finditer("mul\(", line)]
    result = 0
    for idx_start in start_indices:
        idx_end = line.find(")", idx_start)
        if idx_end == -1:
            continue
        inside = line[idx_start + 4 : idx_end]
        parts = inside.split(",")
        if len(parts) != 2:
            continue
        valid_numbers = True
        for part in parts:
            for c in part:
                if c < "0" or c > "9":
                    valid_numbers = False
        if valid_numbers:
            result = result + int(parts[0]) * int(parts[1])
    return result


def day3_part1(fn):
    line, _, _ = read_file_as_single_line(fn)
    return day3_result_from_line(line)


def day3_part2(fn):
    file_line, _, _ = read_file_as_single_line(fn)
    lines = file_line.split("don't()")
    score = day3_result_from_line(lines[0])
    for line in lines[1:]:
        idx_enable = line.find("do()")
        if idx_enable == -1:
            continue
        score = score + day3_result_from_line(line[idx_enable:])
    return score


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


def day6_setup(fn):
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
    obstacle_pos_per_col = [[] for _ in range(line_width)]
    obstacle_pos_per_row = [[] for _ in range(num_lines)]

    for point in points:
        obstacle_pos_per_col[point.x].append(point.y)
        obstacle_pos_per_row[point.y].append(point.x)

    return (
        num_lines,
        line_width,
        points,
        obstacle_pos_per_col,
        obstacle_pos_per_row,
        pos_guard,
        dir,
    )


def day6_move_straight(
    num_lines, line_width, obstacle_pos_per_col, obstacle_pos_per_row, cur_pos, cur_dir
):
    num_steps = 0
    new_pos = Point2i(cur_pos.x, cur_pos.y)
    new_dir = Point2i(cur_dir.x, cur_dir.y)
    max_distance = 1000000
    closest_point_distance = max_distance

    if cur_dir.x != 0:
        for col in obstacle_pos_per_row[cur_pos.y]:
            correct_dir = ((cur_pos.x - col) / cur_dir.x) < 0
            if correct_dir:
                closest_point_distance = abs(cur_pos.x - col)
    if cur_dir.y != 0:
        for row in obstacle_pos_per_col[cur_pos.x]:
            correct_dir = ((cur_pos.y - row) / cur_dir.y) < 0
            if correct_dir:
                closest_point_distance = abs(cur_pos.y - row)

    if closest_point_distance == max_distance:
        if cur_dir.x > 0:
            num_steps = line_width - cur_pos.x - 1
        if cur_dir.x < 0:
            num_steps = cur_pos.x
        if cur_dir.y > 0:
            num_steps = num_lines - cur_pos.y - 1
        if cur_dir.y < 0:
            num_steps = cur_pos.y
        return num_steps, None, None
    num_steps = closest_point_distance - 1
    temp_x = new_dir.x
    new_dir.x = -new_dir.y
    new_dir.y = temp_x
    new_pos.x = cur_pos.x + cur_dir.x * num_steps
    new_pos.y = cur_pos.y + cur_dir.y * num_steps
    return num_steps, new_pos, new_dir


def day6_print_visited(visited, num_lines, line_width):
    for row in range(num_lines):
        for col in range(line_width):
            val = visited[row][col]
            char = "."
            if val:
                char = "x"
            print(char, end="")
        print("")
    print("")


def day6_part1(fn):
    (
        num_lines,
        line_width,
        points,
        obstacle_pos_per_col,
        obstacle_pos_per_row,
        pos_guard,
        cur_dir,
    ) = day6_setup(fn)
    visited = [[False] * line_width for _ in range(num_lines)]
    visited[pos_guard.y][pos_guard.x] = True
    cur_pos = pos_guard
    done = False

    while not done:
        num_steps, new_pos, new_dir = day6_move_straight(
            num_lines,
            line_width,
            obstacle_pos_per_col,
            obstacle_pos_per_row,
            cur_pos,
            cur_dir,
        )
        for idx_step in range(1, num_steps + 1):
            visited[cur_pos.y + cur_dir.y * idx_step][
                cur_pos.x + cur_dir.x * idx_step
            ] = True
        # day6_print_visited(visited, num_lines, line_width)
        if new_pos == None:
            done = True
        else:
            cur_pos.x = new_pos.x
            cur_pos.y = new_pos.y
            cur_dir.x = new_dir.x
            cur_dir.y = new_dir.y

    num_visited = sum(x.count(True) for x in visited)
    return num_visited


def day6_part2(fn):
    (
        num_lines,
        line_width,
        points,
        obstacle_pos_per_col,
        obstacle_pos_per_row,
        pos_guard,
        cur_dir,
    ) = day6_setup(fn)
    dir = [cur_dir]
    pos = [pos_guard]
    cur_pos = pos_guard
    done = False

    while not done:
        num_steps, new_pos, new_dir = day6_move_straight(
            num_lines,
            line_width,
            obstacle_pos_per_col,
            obstacle_pos_per_row,
            cur_pos,
            cur_dir,
        )
        for idx_step in range(1, num_steps + 1):
            visit_dir = Point2i(cur_dir.x, cur_dir.y)
            visit_pos = Point2i(
                cur_pos.x + idx_step * cur_dir.x, cur_pos.y + idx_step * cur_dir.y
            )
            dir.append(visit_dir)
            pos.append(visit_pos)
        if new_pos == None:
            done = True
        else:
            cur_pos.x = new_pos.x
            cur_pos.y = new_pos.y
            cur_dir.x = new_dir.x
            cur_dir.y = new_dir.y

    num_steps = len(dir)
    points.append(Point2i(-1, -1))
    num_loops = 0

    for idx_start in range(num_steps - 1):
        points[-1].x = pos[idx_start + 1].x
        points[-1].y = pos[idx_start + 1].y
        done = False
        stop_dir = []
        stop_pos = []
        cur_pos.x = pos[idx_start].x
        cur_pos.y = pos[idx_start].y
        cur_dir.x = dir[idx_start].x
        cur_dir.y = dir[idx_start].y
        cur_step = 0
        while not done:
            cur_step = cur_step + 1
            num_steps, new_pos, new_dir = day6_move_straight(
                num_lines,
                line_width,
                obstacle_pos_per_col,
                obstacle_pos_per_row,
                cur_pos,
                cur_dir,
            )
            # print("IN ", cur_pos, cur_dir)
            # print(idx_start, num_steps, new_pos, new_dir)
            if new_pos == None:
                done = True
            else:
                if cur_step % 100:
                    for idx_step in range(len(stop_dir)):
                        same_pos = (
                            new_pos.x == stop_pos[idx_step].x
                            and new_pos.y == stop_pos[idx_step].y
                        )
                        same_dir = (
                            new_dir.x == stop_dir[idx_step].x
                            and new_dir.y == stop_dir[idx_step].y
                        )
                        if same_pos and same_dir:
                            num_loops = num_loops + 1
                            if num_loops % 100 == 0:
                                print("Num loops: ", num_loops)
                            done = True
                stop_dir.append(new_dir)
                stop_pos.append(new_pos)
                cur_pos.x = new_pos.x
                cur_pos.y = new_pos.y
                cur_dir.x = new_dir.x
                cur_dir.y = new_dir.y

    # 2425 is too high. 560 too low
    return num_loops


def day7_generate_operators(num_operators, max_num_combinations):
    operators = np.array(
        [[0] * max_num_combinations for _ in range(num_operators**max_num_combinations)]
    )
    print("Shape: ", operators.shape)
    for idx_operator in range(num_operators):
        for col in range(max_num_combinations):
            step_length = num_operators ** (max_num_combinations - col)
            num_el = num_operators ** (max_num_combinations - col - 1)
            num_steps = 1
            for step in range(num_steps):
                row_start = step * step_length + num_el * idx_operator
                row_end_excl = step * step_length + num_el * idx_operator + 1
                print(
                    "To assign: ",
                    row_start,
                    row_end_excl,
                    col,
                    idx_operator,
                )
                operators[row_start:row_end_excl][col] = idx_operator
            # print(step, num_el)
            ## TODO: Generate the "binary" table. Eg for num_operators = 2, max_num_combinations = 3
            ## 000
            ## 001
            ## 010
            ## 011
            ## 100
            ## 101
            ## 110
            ## 111
            ##
            # Num operators = 3, max_num_combinations = 2:
            #
            # 00
            # 01
            # 02
            # 10
            # 11
            # 12
            # 20
            # 21
            # 22
    return operators


def day7_part1(fn):
    # oo = day7_generate_operators(2, 2)
    # print(oo)
    lines = read_file_as_lines(fn)
    score = 0
    max_number_count = 0
    for line in lines:
        parts = line.split(": ")
        the_sum = int(parts[0])
        numbers = list(map(int, parts[1].split(" ")))
        max_number_count = max(max_number_count, len(numbers))
    max_num_operators = max_number_count - 1
    operators = list(map(list, itertools.product([0, 1], repeat=max_num_operators)))
    # TODO: Replace line above with day7_generate_operators()
    for line in lines:
        parts = line.split(": ")
        the_sum = int(parts[0])
        numbers = list(map(int, parts[1].split(" ")))
        found_match = False
        num_operators = len(numbers) - 1
        num_combinations = 2**num_operators
        for combination_whole_line in operators[:num_combinations]:
            combination = combination_whole_line[max_num_operators - num_operators :]
            res = numbers[0]
            for idx, num in enumerate(numbers[1:]):
                if combination[idx] == 0:
                    res = res + num
                else:
                    res = res * num
            if res == the_sum:
                found_match = True
        if found_match:
            score = score + the_sum

    return score


def day7_part2(fn):
    return 0


def day8_hash(row, col):
    return row * 1000 + col


def day8_dehash(val):
    row = val // 1000
    col = val % 1000
    return row, col


def day8_get_element_map(fn):
    lines = read_file_as_lines(fn)
    num_rows = len(lines)
    num_cols = len(lines[0])
    unique_characters = set([el for line in lines for el in line if el != "."])
    element_map = {}
    for c in unique_characters:
        element_map[c] = []
    for row in range(num_rows):
        for col in range(num_cols):
            c = lines[row][col]
            if c != ".":
                element_map[c].append(day8_hash(row, col))
    return element_map, num_rows, num_cols


def day8_part1(fn):
    element_map, num_rows, num_cols = day8_get_element_map(fn)
    antinode_positions = set()
    for _, positions_for_char in element_map.items():
        num_positions = len(positions_for_char)
        for idx_start in range(num_positions):
            row_start, col_start = day8_dehash(positions_for_char[idx_start])
            for idx_end in range(num_positions):
                if idx_start != idx_end:
                    row_end, col_end = day8_dehash(positions_for_char[idx_end])
                    row_antinode = 2 * row_end - row_start
                    col_antinode = 2 * col_end - col_start
                    if row_antinode >= 0 and col_antinode >= 0:
                        if row_antinode < num_rows and col_antinode < num_cols:
                            antinode_positions.add(
                                day8_hash(row_antinode, col_antinode)
                            )
    return len(antinode_positions)


def day8_part2(fn):
    element_map, num_rows, num_cols = day8_get_element_map(fn)
    antinode_positions = set()
    for _, positions_for_char in element_map.items():
        num_positions = len(positions_for_char)
        for idx_start in range(num_positions):
            row_start, col_start = day8_dehash(positions_for_char[idx_start])
            antinode_positions.add(day8_hash(row_start, col_start))
            for idx_end in range(num_positions):
                if idx_start != idx_end:
                    row_end, col_end = day8_dehash(positions_for_char[idx_end])
                    is_valid = True
                    step = 1
                    while is_valid:
                        row_antinode = row_end + (row_end - row_start) * step
                        col_antinode = col_end + (col_end - col_start) * step
                        if row_antinode >= 0 and col_antinode >= 0:
                            if row_antinode < num_rows and col_antinode < num_cols:
                                antinode_positions.add(
                                    day8_hash(row_antinode, col_antinode)
                                )
                            else:
                                is_valid = False
                        else:
                            is_valid = False
                        step += 1
    return len(antinode_positions)


def day9_as_blocks(line):
    blocks = []
    if len(line) % 2 != 0:
        line += "0"
    for _, (block_size, block_space) in enumerate(zip(line[0::2], line[1::2])):
        blocks.append((int(block_size), int(block_space)))
    return blocks


def day9_blocks_and_free(line):
    blocks = []
    free = []
    if len(line) % 2 != 0:
        line += "0"
    idx_pos = 0
    for idx_block, (block_size_string, free_size_string) in enumerate(
        zip(line[::2], line[1::2])
    ):
        block_size = int(block_size_string)
        free_size = int(free_size_string)
        blocks.append((idx_pos, idx_block, block_size))
        idx_pos += block_size
        if free_size > 0:
            free.append((idx_pos, free_size))
            idx_pos += free_size
    return blocks, free


def day9_blocks_to_list(blocks):
    s = []
    for idx, (block_size, block_space) in enumerate(blocks):
        s += [str(idx) for _ in range(block_size)]
        s += ["." for _ in range(block_space)]
    return s


def day9_part1(fn: str):
    line, _, _ = read_file_as_single_line(fn)
    blocks = day9_as_blocks(line)
    blocks_as_list = day9_blocks_to_list(blocks)
    pos_num = []
    pos_space = []
    num_bytes = len(blocks_as_list)
    for idx in range(num_bytes):
        cur_char = blocks_as_list[idx]
        if cur_char == ".":
            pos_space.append(idx)
        else:
            pos_num.append(idx)
    num_num = len(pos_num)
    idx_next_space = 0
    idx_next_num = len(pos_num) - 1
    while pos_space[idx_next_space] < pos_num[idx_next_num]:
        cur_pos_num = pos_num[idx_next_num]
        cur_pos_space = pos_space[idx_next_space]
        cur_num = blocks_as_list[cur_pos_num]
        blocks_as_list[cur_pos_space] = cur_num
        blocks_as_list[cur_pos_num] = "."
        idx_next_space += 1
        idx_next_num -= 1
    res = 0
    for idx in range(num_num):
        res += int(blocks_as_list[idx]) * idx
    return res


def day9_part2(fn: str):
    line, _, _ = read_file_as_single_line(fn)
    blocks, free = day9_blocks_and_free(line)
    res = 0
    for block_pos, block_idx, block_size in blocks[::-1]:
        for free_idx, (free_pos, free_size) in enumerate(free):
            if free_size - block_size >= 0:
                block_pos = free_pos
                free[free_idx] = (free_pos + block_size, free_size - block_size)
                break
        res += block_idx * block_size * (block_pos + (block_pos + block_size - 1)) // 2
    # 8553014718259 too high
    return res


def day10_find(topo, row, col, look_for):
    dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    pos_hash = []
    for row_add, col_add in dirs:
        new_row = row + row_add
        new_col = col + col_add
        if topo[new_row][new_col] == look_for:
            if look_for == 9:
                pos_hash.append(new_row * 1000 + new_col)
            else:
                pos_hash += day10_find(topo, new_row, new_col, look_for + 1)
    return pos_hash


def day10_find_paths(fn: str):
    lines = read_file_as_lines(fn)
    num_rows = len(lines)
    num_cols = len(lines[0])
    lines_as_ints = [[x for x in line] for line in lines]
    topo = np.ones((num_rows + 2, num_cols + 2), dtype=int) * 10
    topo[1 : num_rows + 1, 1 : num_cols + 1] = lines_as_ints
    trailheads = np.argwhere(topo == 0)
    all_hashes = []
    for row, col in trailheads:
        pos_hash = day10_find(topo, row, col, 1)
        all_hashes.append(pos_hash)
    return all_hashes


def day10_part1(fn: str):
    score = 0
    for cur_hash in day10_find_paths(fn):
        score += len(set(cur_hash))
    return score


def day10_part2(fn: str):
    score = 0
    for cur_hash in day10_find_paths(fn):
        score += len(cur_hash)
    return score


def day11_sort(stones, next_stone):
    idx = 0
    done = False
    ret = []
    while not done:
        ret.append(stones[idx])
        idx = next_stone[idx]
        if idx == -1:
            done = True
    return ret


def day11_blink(fn: str, num_blinks):
    line, _, _ = read_file_as_single_line(fn)
    input_stones = [int(s) for s in line.split(" ")]
    max_num_stones = 100000000
    cur_num_stones = len(input_stones)
    stones = np.zeros(max_num_stones, dtype=int)
    next_stone = np.ones(max_num_stones, dtype=int) * -1
    num_digits = np.zeros(max_num_stones, dtype=int)
    stones[0:cur_num_stones] = input_stones
    num_to_left_right = {}
    num_to_multiply = {}
    for idx in range(cur_num_stones - 1):
        next_stone[idx] = idx + 1
    for idx in range(cur_num_stones):
        num_digits[idx] = len(str(stones[idx]))

    for idx in range(num_blinks):
        for idx in range(cur_num_stones):
            cur_num_digits = num_digits[idx]
            if stones[idx] == 0:
                stones[idx] = 1
            elif cur_num_digits % 2 == 0:
                if stones[idx] in num_to_left_right:
                    num_left, num_right, num_dig_left, num_dig_right = (
                        num_to_left_right[stones[idx]]
                    )
                else:
                    as_string = str(stones[idx])
                    num_left = int(as_string[: cur_num_digits // 2])
                    num_right = int(as_string[cur_num_digits // 2 :])
                    num_dig_left = len(str(num_left))
                    num_dig_right = len(str(num_right))
                    num_to_left_right[stones[idx]] = (
                        num_left,
                        num_right,
                        num_dig_left,
                        num_dig_right,
                    )
                stones[cur_num_stones] = num_right
                next_stone[cur_num_stones] = next_stone[idx]
                num_digits[cur_num_stones] = num_dig_right
                stones[idx] = num_left
                next_stone[idx] = cur_num_stones
                num_digits[idx] = num_dig_left
                cur_num_stones += 1
            else:
                if stones[idx] in num_to_multiply:
                    val, num_dig = num_to_multiply[stones[idx]]
                else:
                    val = stones[idx] * 2024
                    num_dig = len(str(val))
                    num_to_multiply[stones[idx]] = (val, num_dig)
                stones[idx] = val
                num_digits[idx] = num_dig

    return cur_num_stones


def day11_splits(fn: str):
    line, _, _ = read_file_as_single_line(fn)
    input_stones = [int(s) for s in line.split(" ")]
    stones = set(input_stones)
    splits = {}
    done = False
    while not done:
        new_stones = set()
        for stone_original in stones:
            if stone_original in splits:
                continue
            is_split = False
            steps = 0
            stone = stone_original
            while is_split == False:
                if stone == 0:
                    stone = 1
                    new_stones.add(stone)
                elif len(str(stone)) % 2 == 0:
                    as_string = str(stone)
                    num_digits = len(as_string)
                    num_left = int(as_string[: num_digits // 2])
                    num_right = int(as_string[num_digits // 2 :])
                    new_stones.add(num_left)
                    new_stones.add(num_right)
                    splits[stone_original] = (steps, num_left, num_right)
                    is_split = True
                else:
                    stone = stone * 2024
                    new_stones.add(stone)
                steps += 1
        len_before = len(stones)
        stones.update(new_stones)
        len_after = len(stones)
        if len_before == len_after:
            done = True
    return splits


def day11_part1(fn: str):
    return day11_blink(fn, 25)


def day11_follow_splits(splits, vals_initial, max_level):
    max_num_vals = 10000000
    vals = np.zeros(max_num_vals, dtype=int)
    vals[: len(vals_initial)] = vals_initial
    levels = np.zeros(max_num_vals, dtype=int)
    idx_cur = len(vals_initial)
    idx_start = 0
    idx_end_excl = len(vals_initial)
    done = False
    num_stones = len(vals_initial)
    while not done:
        idx_cur_old = idx_cur
        for idx in range(idx_start, idx_end_excl):
            steps, val_left, val_right = splits[vals[idx]]
            new_level = levels[idx] + steps + 1
            if new_level > max_level:
                continue
            vals[idx_cur] = val_left
            levels[idx_cur] = new_level
            vals[idx_cur + 1] = val_right
            levels[idx_cur + 1] = new_level
            idx_cur += 2
            num_stones += 1
        num_new = idx_cur - idx_cur_old
        if num_new == 0:
            done = True
        else:
            idx_start = idx_end_excl
            idx_end_excl = idx_start + num_new
    return num_stones


def day11_part2(fn: str):
    line, _, _ = read_file_as_single_line(fn)
    input_stones = [int(s) for s in line.split(" ")]
    splits = day11_splits(fn)
    num_stones = 0
    max_level = 25
    num_stones = day11_follow_splits(splits, input_stones, max_level)
    return num_stones


def day12_floodfill(garden, is_taken, num_rows, num_cols, start_row, start_col) -> int:
    # Segment map has True for each element that is part of the current segment
    # Pad the segment map with one element in each direction so we do not have
    #   have to check for out of bounds later on
    segment_map = [[False] * (num_cols + 2) for _ in range(num_rows + 2)]
    pos_to_check = []
    pos_to_check.append((start_row, start_col))
    is_taken[start_row][start_col] = True
    idx_start = 0
    num_elements = len(pos_to_check)
    plot_type = garden[start_row][start_col]
    area = 1
    num_perimeter_plots = 0
    done = False
    while not done:
        idx_end_excl = idx_start + num_elements
        num_elements = 0
        for idx_el in range(idx_start, idx_end_excl):
            cur_pos_row = pos_to_check[idx_el][0]
            cur_pos_col = pos_to_check[idx_el][1]
            for idx_new in range(4):
                delta_col = [-1, 0, 1, 0][idx_new]
                delta_row = [0, -1, 0, 1][idx_new]
                new_row = cur_pos_row + delta_row
                new_col = cur_pos_col + delta_col
                outside_col = new_col < 0 or new_col >= num_cols
                outside_row = new_row < 0 or new_row >= num_rows
                if outside_col or outside_row:
                    num_perimeter_plots = num_perimeter_plots + 1
                    continue
                cur_plot_type = garden[new_row][new_col]
                segment_map[cur_pos_row + 1][cur_pos_col + 1] = True
                if cur_plot_type == plot_type:
                    if not is_taken[new_row][new_col]:
                        is_taken[new_row][new_col] = True
                        area = area + 1
                        pos_to_check.append((new_row, new_col))
                        num_elements = num_elements + 1
                else:
                    num_perimeter_plots = num_perimeter_plots + 1
        if num_elements == 0:
            done = True
        idx_start = idx_end_excl

    return area, num_perimeter_plots, segment_map


def day12_part1(fn):
    garden = read_file_as_lines(fn)
    num_rows = len(garden)
    num_cols = len(garden[0])
    is_taken = [[False] * num_cols for _ in range(num_rows)]

    total_score = 0

    for row in range(num_rows):
        for col in range(num_cols):
            if is_taken[row][col] == False:
                area, num_perimeter_plots, _ = day12_floodfill(
                    garden, is_taken, num_rows, num_cols, row, col
                )
                total_score = total_score + area * num_perimeter_plots
    return total_score


def day12_part2(fn):
    garden = read_file_as_lines(fn)
    num_rows = len(garden)
    num_cols = len(garden[0])
    is_taken = [[False] * num_cols for _ in range(num_rows)]

    total_score = 0

    for row in range(num_rows):
        for col in range(num_cols):
            if is_taken[row][col] == False:
                area, _, segment_map = day12_floodfill(
                    garden, is_taken, num_rows, num_cols, row, col
                )
                num_edges = 0
                for idx_direction in range(2):
                    # Rotate the second time around, so we count vertical edges as well
                    if idx_direction == 1:
                        for row_rotate in range(0, num_rows + 2):
                            for col_rotate in range(0, num_rows + 2):
                                row_rotate_t = col_rotate
                                col_rotate_t = row_rotate
                                temp = segment_map[row_rotate][col_rotate]
                                segment_map[row_rotate][col_rotate] = segment_map[
                                    row_rotate_t
                                ][col_rotate_t]
                                segment_map[row_rotate_t][col_rotate_t] = temp
                    for row_perimeter in range(1, num_rows + 2):
                        last_type = 0
                        for col_perimeter in range(0, num_cols + 2):
                            plot_above = segment_map[row_perimeter - 1][col_perimeter]
                            plot_this = segment_map[row_perimeter][col_perimeter]
                            this_type = 0
                            if plot_above and not plot_this:
                                this_type = 1
                            if not plot_above and plot_this:
                                this_type = 2
                            if (this_type != last_type) and this_type != 0:
                                num_edges = num_edges + 1
                            last_type = this_type
                total_score = total_score + area * num_edges
    return total_score


def day13_get_machines(fn: str):
    lines = read_file_as_lines(fn)
    num_machines = (len(lines) + 1) // 4
    btn_a = []
    btn_b = []
    price = []
    for idx in range(num_machines):
        line_a = lines[4 * idx]
        line_b = lines[4 * idx + 1]
        line_price = lines[4 * idx + 2]
        a_x, a_y = line_a.split("Button A: X+")[1].split(", Y+")
        b_x, b_y = line_b.split("Button B: X+")[1].split(", Y+")
        p_x, p_y = line_price.split("Prize: X=")[1].split(", Y=")
        btn_a.append((int(a_x), int(a_y)))
        btn_b.append((int(b_x), int(b_y)))
        price.append((int(p_x), int(p_y)))
    return num_machines, btn_a, btn_b, price


def day13_part1(fn: str):
    num_machines, btn_a, btn_b, price = day13_get_machines(fn)
    total_cost = 0
    for idx, (cur_a, cur_b, cur_price) in enumerate(zip(btn_a, btn_b, price)):
        ax, ay = cur_a
        bx, by = cur_b
        px, py = cur_price
        # Solve a simple linear system with two variables
        # 1. Multiply first line with "x" variable of second line
        #   so we can remove the A variable from the equation
        bx *= ay
        px *= ay
        by *= ax
        py *= ax
        # 2. Subtract second equation from first equation
        bx -= by
        px -= py
        # 3. Calculate B
        num_b_real = px / bx
        # 4. Calculate A using B
        ax, ay = cur_a
        bx, by = cur_b
        px, py = cur_price
        num_a_real = (px - bx * num_b_real) / ax
        num_a = int(num_a_real)
        num_b = int(num_b_real)
        num_err_tolerance = 0.000001
        x_is_int = abs(num_a_real - num_a) < num_err_tolerance
        y_is_int = abs(num_b_real - num_b) < num_err_tolerance
        is_valid = x_is_int and y_is_int
        if is_valid:
            cost = 3 * num_a + 1 * num_b
            total_cost += cost
    return total_cost


def day17_read_input(fn):
    lines = read_file_as_lines(fn)
    registers = []
    for line in lines:
        is_register = line.find("Register") != -1
        is_program = line.find("Program") != -1
        parts = line.split(": ")
        if is_register:
            registers.append(int(parts[1]))
        if is_program:
            numbers_str = parts[1].split(",")
            numbers = list(map(lambda x: int(x), numbers_str))
            op_list = list(zip(numbers[::2], numbers[1::2]))
    return registers, op_list


def day17_run_program(registers, cmd, literal):
    new_idx_op = None
    new_output_val = None
    combo = 0
    if literal <= 3:
        combo = literal
    elif combo <= 6:
        combo = registers[literal - 4]
    match cmd:
        case 0:
            registers[0] = registers[0] >> combo
        case 1:
            registers[1] = registers[1] ^ literal
        case 2:
            registers[1] = combo % 8
        case 3:
            if registers[0] != 0:
                new_idx_op = literal
        case 4:
            registers[1] = registers[1] ^ registers[2]
        case 5:
            new_output_val = combo % 8
        case 6:
            registers[1] = registers[0] >> combo
        case 7:
            registers[2] = registers[0] >> combo
    return new_idx_op, new_output_val


def day17_part1(fn):
    registers, op_list = day17_read_input(fn)
    done = False
    idx_op = 0
    output_vals = []
    while not done:
        cmd, literal = op_list[idx_op]
        new_idx_op, new_output_val = day17_run_program(registers, cmd, literal)
        if new_idx_op != None:
            idx_op = new_idx_op
            continue
        if new_output_val != None:
            output_vals.append(new_output_val)
        idx_op = idx_op + 1
        if idx_op >= len(op_list):
            done = True
    res = ",".join(str(n) for n in output_vals)
    return res


def day17_part2(fn):
    registers, op_list = day17_read_input(fn)
    res = 0
    program = [val for op in op_list for val in op]
    print("Program: ", program)
    num_levels = len(program)

    # Output is 3-bit, meaning we have 7 overlapping bits between input of two
    #   adjacent outputs
    output_to_input = [[] for _ in range(8)]
    for a_val in range(1024):
        registers[0] = a_val
        for cmd, literal in op_list:
            _, new_output_val = day17_run_program(registers, cmd, literal)
            if new_output_val != None:
                output_to_input[new_output_val].append(a_val)
    combos_per_level = [[] for _ in range(num_levels)]
    [combos_per_level[0].append([x]) for x in output_to_input[program[-1]]]
    for idx_level in range(1, num_levels):
        desired_output = program[-idx_level - 1]
        for low_list in combos_per_level[idx_level - 1]:
            val_low = low_list[-1]
            for val_high in output_to_input[desired_output]:
                if ((val_low >> 3) & 0b1111111) == (val_high & 0b1111111):
                    combos_per_level[idx_level].append(low_list + [val_high])
        print(
            f"Ran level {idx_level} with desired output {desired_output}. Got {len(combos_per_level[idx_level])} combos"
        )

    all_possible_vals = []
    for ok_combos in combos_per_level[-1]:
        the_val = 0
        for idx_level in range(num_levels):
            the_val |= ok_combos[idx_level] << (idx_level * 3)
        all_possible_vals.append(the_val)

    all_possible_vals.sort()
    smallest = all_possible_vals[0]
    return smallest


def aoc_2024(run_real=True, single_day=None):
    day_start = 1
    day_end_excl = 26
    if single_day != None:
        day_start = single_day
        day_end_excl = single_day + 1
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


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        single_day = int(sys.argv[1])
        aoc_2024(single_day=single_day)
    else:
        aoc_2024()
