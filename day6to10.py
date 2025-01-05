import itertools
import numpy as np
import math
from dataclasses import dataclass
from shared import *


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
