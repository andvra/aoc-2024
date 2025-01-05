import math
import sys
import numpy as np
from typing import List
import copy
import time
from shared import *
from first10 import *


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


def day13_solve(a, b, price, val_to_add=0):
    ax, ay = a
    bx, by = b
    px, py = price
    px += val_to_add
    py += val_to_add
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
    ax, ay = a
    bx, by = b
    px, py = price
    px += val_to_add
    py += val_to_add
    num_a_real = (px - bx * num_b_real) / ax
    num_a = int(num_a_real)
    num_b = int(num_b_real)
    num_err_tolerance = 0.000001
    x_is_int = abs(num_a_real - num_a) < num_err_tolerance
    y_is_int = abs(num_b_real - num_b) < num_err_tolerance
    is_valid = x_is_int and y_is_int
    return is_valid, num_a, num_b


def day13_part1(fn: str):
    num_machines, btn_a, btn_b, price = day13_get_machines(fn)
    total_cost = 0
    for idx, (cur_a, cur_b, cur_price) in enumerate(zip(btn_a, btn_b, price)):
        is_valid, num_a, num_b = day13_solve(cur_a, cur_b, cur_price)
        if is_valid:
            cost = 3 * num_a + 1 * num_b
            total_cost += cost
    return total_cost


def day13_part2(fn: str):
    num_machines, btn_a, btn_b, price = day13_get_machines(fn)
    total_cost = 0
    for idx, (cur_a, cur_b, cur_price) in enumerate(zip(btn_a, btn_b, price)):
        is_valid, num_a, num_b = day13_solve(cur_a, cur_b, cur_price, 10000000000000)
        if is_valid:
            cost = 3 * num_a + 1 * num_b
            total_cost += cost
    return total_cost


def day14_get_input(fn: str):
    lines = read_file_as_lines(fn)
    positions = []
    velocities = []
    for line in lines:
        pos, vel = line.split("p=")[1].split(" v=")
        pos_col_s, pos_row_s = pos.split(",")
        vel_col_s, vel_row_s = vel.split(",")
        positions.append((int(pos_row_s), int(pos_col_s)))
        velocities.append((int(vel_row_s), int(vel_col_s)))
    return len(lines), positions, velocities


def day14_as_image(rows, cols, pos):
    image = [["."] * cols for _ in range(rows)]
    for row, col in pos:
        image[row][col] = "P"
    image = image[::2][::2]
    return image


def day14_part1(fn: str):
    num_robots, pos, vel = day14_get_input(fn)
    rows, cols = 103, 101
    num_steps = 100
    for step in range(num_steps):
        for idx_robot in range(num_robots):
            new_row = (pos[idx_robot][0] + vel[idx_robot][0] + rows) % rows
            new_col = (pos[idx_robot][1] + vel[idx_robot][1] + cols) % cols
            pos[idx_robot] = (new_row, new_col)
    num_in_quad = [0, 0, 0, 0]
    half_cols = cols // 2
    half_rows = rows // 2
    quad_start = [
        (0, 0),
        (0, half_cols + 1),
        (half_rows + 1, 0),
        (half_rows + 1, half_cols + 1),
    ]
    quad_end = [
        (half_rows - 1, half_cols - 1),
        (half_rows - 1, cols - 1),
        (rows - 1, half_cols - 1),
        (rows - 1, cols - 1),
    ]
    for cur_pos in pos:
        prow, pcol = cur_pos
        for idx in range(4):
            srow, scol = quad_start[idx]
            erow, ecol = quad_end[idx]
            ok_row = prow >= srow and prow <= erow
            ok_col = pcol >= scol and pcol <= ecol
            if ok_row and ok_col:
                num_in_quad[idx] += 1

    return math.prod(num_in_quad)


def day14_part2(fn: str):
    if fn.find("test") > -1:
        return -1
    num_robots, pos, vel = day14_get_input(fn)
    rows, cols = 103, 101
    num_steps = 100000
    num_buckets = 10
    num_vals = np.zeros((num_buckets, num_buckets), dtype=int)
    for step in range(num_steps):
        num_vals[:, :] = 0
        for idx_robot in range(num_robots):
            new_row = (pos[idx_robot][0] + vel[idx_robot][0] + rows) % rows
            new_col = (pos[idx_robot][1] + vel[idx_robot][1] + cols) % cols
            pos[idx_robot] = (new_row, new_col)
            if new_row < 100 and new_col < 100:
                num_vals[new_row // 10][new_col // 10] += 1
        the_small = np.where(num_vals[:, :] < 3)
        num_small = len(the_small[0])
        # NB number below, as well as limit for low noise area, is defined after testing
        if num_small > 60:
            image = day14_as_image(rows, cols, pos)
            for row in image:
                for col in row:
                    print(col, end="")
                print()
            input(f"Steps for image above: {step+1}")


def day15_get_data(fn: str):
    lines = read_file_as_lines(fn)
    lines_board = []
    lines_move = []
    add_to_board = True
    for line in lines:
        if line == "":
            add_to_board = False
        else:
            if add_to_board:
                lines_board.append(line)
            else:
                lines_move.append(line)
    num_rows = len(lines_board)
    num_cols = len(lines_board[0])
    walls = []
    boxes = []
    pos = (0, 0)
    for row in range(num_rows):
        for col in range(num_cols):
            char = lines_board[row][col]
            cur_pos = (row, col)
            if char == "@":
                pos = cur_pos
            elif char == "#":
                walls.append(cur_pos)
            elif char == "O":
                boxes.append(cur_pos)
    line_moves = "".join(lines_move)
    moves = []
    for c in line_moves:
        if c == "<":
            moves.append((0, -1))
        if c == ">":
            moves.append((0, 1))
        if c == "v":
            moves.append((1, 0))
        if c == "^":
            moves.append((-1, 0))
    return num_rows, num_cols, walls, boxes, moves, pos


def day15_part1(fn: str):
    num_rows, num_cols, walls, boxes, moves, pos = day15_get_data(fn)
    board = [["."] * num_cols for _ in range(num_rows)]
    for row, col in walls:
        board[row][col] = "#"
    for row, col in boxes:
        board[row][col] = "O"
    for row_move, col_move in moves:
        cur_row, cur_col = pos
        new_row = cur_row + row_move
        new_col = cur_col + col_move
        if board[new_row][new_col] == ".":
            pos = (new_row, new_col)
        elif board[new_row][new_col] == "O":
            can_move = False
            try_row = 0
            try_col = 0
            for idx in range(1, max(num_rows, num_cols)):
                try_row = new_row + row_move * idx
                try_col = new_col + col_move * idx
                if board[try_row][try_col] == "#":
                    break
                if board[try_row][try_col] == ".":
                    can_move = True
                    break
            if can_move:
                board[try_row][try_col] = "O"
                board[new_row][new_col] = "."
                pos = (new_row, new_col)
    squares = np.where(np.array(board) == "O")
    res = 0
    for row, col in zip(squares[0], squares[1]):
        res += 100 * row + col
    return res


def day15_print_board(board, pos):
    board_to_print = copy.deepcopy(board)
    pos_row, pos_col = pos
    board_to_print[pos_row][pos_col] = "@"
    for line in board_to_print:
        print("".join(line))


def day15_part2(fn: str):
    num_rows, num_cols_orig, walls, boxes, moves, pos_orig = day15_get_data(fn)
    pos_row, pos_orig_col = pos_orig
    pos = (pos_row, pos_orig_col * 2)
    num_cols = num_cols_orig * 2
    board = [["."] * num_cols for _ in range(num_rows)]
    for row, col_half in walls:
        col_start = col_half * 2
        board[row][col_start : col_start + 2] = "##"
    for row, col_half in boxes:
        col_start = col_half * 2
        board[row][col_start : col_start + 2] = "[]"

    for idx_move, (row_move, col_move) in enumerate(moves):
        row_cur, col_cur = pos
        char_cur = board[row_cur + row_move][col_cur + col_move]
        if char_cur == ".":
            pos = (row_cur + row_move, col_cur + col_move)
        elif char_cur in ["[", "]"]:
            if col_move != 0:
                row_test = row_cur
                for idx_test in range(1, max(num_rows, num_cols)):
                    col_test = col_cur + idx_test * col_move
                    cur_char = board[row_test][col_test]
                    if cur_char == ".":
                        for idx in range(idx_test):
                            col = col_test - idx * col_move
                            board[row_test][col] = board[row_test][col - col_move]
                        pos = (row_cur, col_cur + col_move)
                        break
                    elif cur_char == "#":
                        break
            if row_move != 0:
                # Moving up/down is performed in two steps:
                #   1. Find boxes that are connected. We register the leftmost piece of each box
                #   2. For each box, check if it can move
                if char_cur == "[":
                    con_boxes = [(row_cur + row_move, col_cur)]
                else:
                    con_boxes = [(row_cur + row_move, col_cur - 1)]
                idx_start = 0
                idx_end_excl = len(con_boxes)
                done = False
                while not done:
                    new_con = []
                    for idx_con in range(idx_start, idx_end_excl):
                        row_con, col_con = con_boxes[idx_con]
                        char_same_col = board[row_con + row_move][col_con]
                        char_next_col = board[row_con + row_move][col_con + 1]
                        if char_same_col == "[":
                            new_con.append((row_con + row_move, col_con))
                        if char_same_col == "]":
                            new_con.append((row_con + row_move, col_con - 1))
                        if char_next_col == "[":
                            new_con.append((row_con + row_move, col_con + 1))
                    if len(new_con) > 0:
                        new_con = list(dict.fromkeys(new_con))
                        for n in new_con:
                            con_boxes.append(n)
                        idx_start = idx_end_excl
                        idx_end_excl = idx_start + len(new_con)
                    else:
                        done = True
                can_move = True
                for row, col in con_boxes:
                    cur_pieces = board[row + row_move][col : col + 2]
                    if "#" in cur_pieces:
                        can_move = False
                if can_move:
                    for row, col in con_boxes[::-1]:
                        board[row + row_move][col : col + 2] = "[]"
                        board[row][col : col + 2] = ".."
                    pos = (row_cur + row_move, col_cur)
    squares = np.where(np.array(board) == "[")
    res = 0
    for row, col in zip(squares[0], squares[1]):
        res += 100 * row + col
    return res


def day16_pos_to_hash(row, col, new_row_move, new_col_move):
    idx_rotation = 0
    if new_row_move != 0:
        idx_rotation = (new_row_move + 1) // 2
    if new_col_move != 0:
        idx_rotation = ((new_col_move + 1) // 2) + 2
    return 10000 * row + 10 * col + idx_rotation


def day16_unique_squares(head_parents, heads, idx_first):
    squares = set()
    idx_search = [idx_first]
    idx_start = 0
    idx_end_excl = len(idx_search)
    done = False
    while not done:
        num_new = 0
        for idx in range(idx_start, idx_end_excl):
            idx_head = idx_search[idx]
            if not idx_head in head_parents:
                done = True
                break
            row, col, _, _, _ = heads[idx_head]
            for idx_parent in head_parents[idx_head]:
                row_parent, col_parent, row_move, col_move, _ = heads[idx_parent]
                for iter in range(abs(row_parent - row) + abs(col_parent - col) + 1):
                    new_row = row_parent + iter * row_move
                    new_col = col_parent + iter * col_move
                    new_hash = new_row * 1000 + new_col
                    squares.add(new_hash)
                num_new += 1
                idx_search.append(idx_parent)
        if num_new == 0:
            done = True
        else:
            idx_start = idx_end_excl
            idx_end_excl = idx_start + num_new
    return squares


def day16_move(fn: str):
    board = read_file_as_lines(fn)
    num_rows = len(board)
    num_cols = len(board[0])
    flat = "".join(board)
    pos_s = flat.find("S")
    pos_e = flat.find("E")
    pos_start = (pos_s // num_rows, pos_s % num_cols)
    pos_end = (pos_e // num_rows, pos_e % num_cols)
    max_num_heads = 10000
    heads = [()] * max_num_heads
    head_hash = [0] * max_num_heads
    heads[0] = pos_start + (0, 1, 0)
    head_hash[0] = day16_pos_to_hash(pos_start[0], pos_start[1], 0, 1)
    num_heads = 1
    if board[pos_start[0] - 1][pos_start[1]] == ".":
        heads[1] = pos_start + (-1, 0, 1000)
        head_hash[1] = day16_pos_to_hash(pos_start[0], pos_start[1], -1, 0)
        num_heads += 1
    used = [0] * max_num_heads
    best_head = {}
    head_parents = {}
    done = False
    best_score = 100000000
    best_score_heads = []
    old_scores = {}
    while not done:
        idx_best = -1
        cur_best_score = 1000000000
        for idx_used in range(num_heads):
            if used[idx_used] == 0:
                _, _, _, _, cur_score = heads[idx_used]
                if cur_score < cur_best_score:
                    idx_best = idx_used
                    cur_best_score = cur_score
        if idx_best == -1:
            done = True
        else:
            row, col, row_move, col_move, score = heads[idx_best]
            left_row_move = -col_move
            left_col_move = row_move
            right_row_move = -left_row_move
            right_col_move = -left_col_move
            for _ in range(1, num_rows):
                row += row_move
                col += col_move
                score += 1
                if board[row][col] == "#":
                    break
                if board[row][col] == "E":
                    if score <= best_score:
                        hash_new = day16_pos_to_hash(row, col, row_move, col_move)
                        new_head = (row, col, row_move, col_move, score)
                        heads[num_heads] = new_head
                        used[num_heads] = 1
                        best_head[hash_new] = num_heads
                        head_parents[num_heads] = [idx_best]
                        if score == best_score:
                            best_score_heads.append(num_heads)
                        elif score < best_score:
                            best_score_heads.clear()
                            best_score_heads.append(num_heads)
                            best_score = score
                        num_heads += 1
                    break
                move_to_add = []
                if board[row + left_row_move][col + left_col_move] == ".":
                    move_to_add.append((left_row_move, left_col_move))
                if board[row + right_row_move][col + right_col_move] == ".":
                    move_to_add.append((right_row_move, right_col_move))
                for row_move_add, col_move_add in move_to_add:
                    hash_new = day16_pos_to_hash(row, col, row_move_add, col_move_add)
                    new_score = score + 1000
                    if (hash_new in old_scores) and (old_scores[hash_new] == new_score):
                        idx_head = best_head[hash_new]
                        head_parents[idx_head].append(idx_best)
                    elif (hash_new not in old_scores) or old_scores[
                        hash_new
                    ] > new_score:
                        new_head = (row, col, row_move_add, col_move_add, new_score)
                        heads[num_heads] = new_head
                        used[num_heads] = 0
                        best_head[hash_new] = num_heads
                        head_parents[num_heads] = [idx_best]
                        old_scores[hash_new] = new_score
                        num_heads += 1
            used[idx_best] = 1
    return best_score, best_score_heads, head_parents, heads, board


def day16_part1(fn: str):
    best_score, _, _, _, _ = day16_move(fn)
    return best_score


def day16_part2(fn: str):
    _, best_score_heads, head_parents, heads, board = day16_move(fn)
    all_squares = set()
    for idx_head in best_score_heads:
        all_squares.update(day16_unique_squares(head_parents, heads, idx_head))
    return len(all_squares)


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


def day18_print_grid(grid_size, coords, num_coords):
    chars = [["."] * grid_size for _ in range(grid_size)]
    for row, col in coords[:num_coords]:
        chars[row][col] = "#"
    for line in chars:
        print("".join(line))


def day18_part1(fn: str):
    lines = read_file_as_lines(fn)
    coords = []
    for line in lines:
        parts = line.split(",")
        # We are adding padding to the board, so coordinates
        #   needs to be updated
        coords.append((int(parts[1]) + 1, int(parts[0]) + 1))
    grid_size = 7
    num_coords_to_use = 12
    if fn.find("real") > -1:
        grid_size = 71
        num_coords_to_use = 1024
    # day18_print_grid(grid_size, coords, 12)
    pos_start = (1, 1)
    pos_end = (grid_size, grid_size)
    # Adding padding
    grid = [[(-1, -1)] * (grid_size + 2) for _ in range(grid_size + 2)]
    for row in range(1, grid_size + 1):
        for col in range(1, grid_size + 1):
            grid[row][col] = (1000000, 1000000)
    for row, col in coords[:num_coords_to_use]:
        grid[row][col] = (-1, -1)
    to_expand_start = set()
    to_expand_end = set()
    to_expand_start.add(pos_start)
    to_expand_end.add(pos_end)
    done = False
    steps_shortest = 1000000
    for idx_step in range(1, 1000000):
        num_added = 0
        done = False
        for idx_set in range(2):
            if idx_set == 0:
                cur_set = to_expand_start
            else:
                cur_set = to_expand_end
            add_to_set = set()
            for row, col in cur_set:
                for row_add, col_add in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_row = row + row_add
                    new_col = col + col_add
                    cur_score = grid[new_row][new_col]
                    if idx_step < cur_score[idx_set]:
                        cur_score = (
                            cur_score[:idx_set] + (idx_step,) + cur_score[idx_set + 1 :]
                        )
                        grid[new_row][new_col] = cur_score
                        add_to_set.add((new_row, new_col))
                        if cur_score[(idx_set + 1) % 2] < 1000000:
                            steps_shortest = min(steps_shortest, sum(cur_score))
                            done = True
            num_added += len(add_to_set)
            cur_set.update(add_to_set)
        if num_added == 0:
            done = True
        if done:
            break
    return steps_shortest


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
