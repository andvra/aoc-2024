from functools import cache
from shared import *


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
    # print("Program: ", program)
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
        # print(
        #     f"Ran level {idx_level} with desired output {desired_output}. Got {len(combos_per_level[idx_level])} combos"
        # )

    all_possible_vals = []
    for ok_combos in combos_per_level[-1]:
        the_val = 0
        for idx_level in range(num_levels):
            the_val |= ok_combos[idx_level] << (idx_level * 3)
        all_possible_vals.append(the_val)

    all_possible_vals.sort()
    smallest = all_possible_vals[0]
    return smallest


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
    grid[pos_start[0]][pos_start[1]] = (0, 1000000)
    grid[pos_end[0]][pos_end[1]] = (1000000, 0)
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


def day18_add_coord(coords, grid, segments, coord_segment, grid_size, idx_new_coord):
    add_new_segment = True
    row, col = coords[idx_new_coord]
    con_above = row == 1
    con_below = row == grid_size
    con_left = col == 1
    con_right = col == grid_size
    new_con = (con_above, con_right, con_below, con_left)
    idx_segment = -1
    found_segment_ids = set()
    for idx_grid in range(9):
        row_add = idx_grid // 3 - 1
        col_add = idx_grid % 3 - 1
        row_eval = row + row_add
        col_eval = col + col_add
        idx_existing_coord = grid[row_eval][col_eval]
        if idx_existing_coord > -1:
            idx_segment = coord_segment[idx_existing_coord]
            found_segment_ids.add(idx_segment)
    if len(found_segment_ids) > 0:
        idx_segment = min(found_segment_ids)
        for idx_merge_segment in found_segment_ids:
            combo = tuple(
                x or y
                for x, y in zip(segments[idx_segment], segments[idx_merge_segment])
            )
            segments[idx_segment] = combo
        coord_segment[:] = [
            idx_segment if x in found_segment_ids else x for x in coord_segment
        ]
    if idx_segment > -1:
        coord_segment.append(idx_segment)
        if sum(new_con) > 0:
            combo = tuple(x or y for x, y in zip(segments[idx_segment], new_con))
            segments[idx_segment] = combo
        add_new_segment = False
    if add_new_segment:
        segments.append(new_con)
        coord_segment.append(len(segments) - 1)
    grid[row][col] = idx_new_coord


def day18_part2(fn: str):
    lines = read_file_as_lines(fn)
    coords = []
    for line in lines:
        parts = line.split(",")
        coords.append((int(parts[1]) + 1, int(parts[0]) + 1))
    segments = []
    grid_size = 7
    num_initial_coords = 12
    if fn.find("real") > -1:
        num_initial_coords = 1024
        grid_size = 71
    grid = [[-1] * (grid_size + 2) for _ in range(grid_size + 2)]
    coord_segment = []
    for idx_new_coord in range(num_initial_coords):
        day18_add_coord(coords, grid, segments, coord_segment, grid_size, idx_new_coord)
    failed_coord = (-1, -1)
    for idx_new_coord in range(num_initial_coords, len(coords)):
        day18_add_coord(coords, grid, segments, coord_segment, grid_size, idx_new_coord)
        con = False
        for con_above, con_right, con_below, con_left in segments:
            con = con or (con_above and con_left)
            con = con or (con_above and con_below)
            con = con or (con_below and con_right)
            con = con or (con_left and con_right)
        if con:
            row_adj, col_adj = coords[idx_new_coord]
            failed_coord = (col_adj - 1, row_adj - 1)
            break
    return failed_coord


def day19_part1(fn: str):
    lines = read_file_as_lines(fn)
    towels = set(lines[0].split(", "))
    designs = lines[2:]

    @cache
    def check_possible(design: str):
        if len(design) == 0:
            return True
        found_ok = False
        for towel in towels:
            found_ok = found_ok or (
                design.startswith(towel) and check_possible(design[len(towel) :])
            )
        return found_ok

    return sum(map(check_possible, designs))


def day19_part2(fn: str):
    lines = read_file_as_lines(fn)
    towels = set(lines[0].split(", "))
    designs = lines[2:]

    @cache
    def all_combos(design: str):
        if len(design) == 0:
            return 1
        num_combos = 0
        for towel in towels:
            if design.startswith(towel):
                num_combos += all_combos(design[len(towel) :])
        return num_combos

    return sum(map(all_combos, designs))


def day20_read_board(fn: str):
    lines = read_file_as_lines(fn)
    board = []
    for line in lines:
        board.append(list(line))
    num_rows = len(board)
    num_cols = len(board[0])
    for row in range(num_rows):
        for col in range(num_cols):
            if board[row][col] == "S":
                pos_start = (row, col)
            if board[row][col] == "E":
                pos_end = (row, col)
    board[pos_start[0]][pos_start[1]] = "."
    board[pos_end[0]][pos_end[1]] = "."
    return board, num_rows, num_cols, pos_start, pos_end


def day20_solve_board(board, pos_start, pos_end):
    board[pos_start[0]][pos_start[1]] = 0
    cur_pos = pos_start
    while cur_pos != pos_end:
        row, col = cur_pos
        cur_val = board[row][col]
        for row_move, col_move in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row_eval = row + row_move
            col_eval = col + col_move
            if board[row_eval][col_eval] == ".":
                board[row_eval][col_eval] = cur_val + 1
                cur_pos = (row_eval, col_eval)
                break


def day20_part1(fn: str):
    board, num_rows, num_cols, pos_start, pos_end = day20_read_board(fn)
    row_shortcuts = []
    col_shortcuts = []
    for row in range(1, num_rows - 1):
        for col in range(1, num_cols - 1):
            row_slice = [board[idx][col] for idx in range(row - 1, row + 2)]
            if row_slice == [".", "#", "."]:
                row_shortcuts.append((row, col))
            col_slice = [board[row][idx] for idx in range(col - 1, col + 2)]
            if col_slice == [".", "#", "."]:
                col_shortcuts.append((row, col))
    day20_solve_board(board, pos_start, pos_end)
    shortcuts = {}
    for row, col in row_shortcuts:
        reward = abs(board[row + 1][col] - board[row - 1][col]) - 2
        shortcuts[reward] = shortcuts.get(reward, 0) + 1
    for row, col in col_shortcuts:
        reward = abs(board[row][col + 1] - board[row][col - 1]) - 2
        shortcuts[reward] = shortcuts.get(reward, 0) + 1
    return sum([v for k, v in shortcuts.items() if k >= 100])


def day20_part2(fn: str):
    board, num_rows, num_cols, pos_start, pos_end = day20_read_board(fn)
    day20_solve_board(board, pos_start, pos_end)
    max_steps = 20
    min_cheat = 50

    if fn.find("real") > -1:
        min_cheat = 100

    def find_cheats(start_score, pos):
        cheats = set()
        heads = [pos]
        idx_start = 0
        idx_end_excl = len(heads)
        num_steps = 0
        visited = set()
        visited.update(heads)
        while num_steps <= max_steps:
            num_new = 0
            for idx in range(idx_start, idx_end_excl):
                row, col = heads[idx]
                if isinstance(board[row][col], int):
                    end_score = board[row][col]
                    new_cheat = end_score - start_score - num_steps
                    if new_cheat >= min_cheat:
                        cheats.add((start_score, end_score, new_cheat, num_steps))
                for row_add, col_add in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    row_eval = row + row_add
                    col_eval = col + col_add
                    if row_eval < 0 or row_eval >= num_rows:
                        continue
                    if col_eval < 0 or col_eval >= num_cols:
                        continue
                    new_pos = (row_eval, col_eval)
                    if new_pos in visited:
                        continue
                    heads.append(new_pos)
                    visited.add(new_pos)
                    num_new += 1

            num_steps += 1
            idx_start = idx_end_excl
            idx_end_excl = idx_start + num_new
        return cheats

    all_cheats = set()
    for row in range(num_rows):
        for col in range(num_cols):
            if isinstance(board[row][col], int):
                start_score = board[row][col]
                pos = (row, col)
                all_cheats.update(find_cheats(start_score, pos))
    scores = {}
    for _, _, score, _ in all_cheats:
        scores[score] = scores.get(score, 0) + 1
    scores = dict(sorted(scores.items(), key=lambda item: item[0]))
    res = sum(v for _, v in scores.items())

    return res
