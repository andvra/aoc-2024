from shared import *


def day21_part1(fn: str):
    if fn.find("real") > -1:
        return -1
    lines = read_file_as_lines(fn)
    codes = []
    for line in lines:
        code = list(line)
        code[-1] = "10"
        code = list(map(lambda x: int(x), code))
        codes.append(code)
    # Key A is number 10 (= idx 11) on numeric
    pos_numeric = {}
    pos_numeric[0] = (3, 1)
    for n in range(1, 10):
        row = 2 - (n - 1) // 3
        col = (n - 1) % 3
        pos_numeric[n] = (row, col)
    pos_numeric[10] = (3, 2)
    numeric_shortest = {}
    for idx in range(11):
        numeric_shortest[idx] = {}

    def find_paths(positions, indices, set_shortest):
        all_pos = set()
        for idx in indices:
            all_pos.add(positions[idx])
        for idx_end_offset, idx_start in enumerate(indices):
            row_start, col_start = positions[idx_start]
            for idx_end in indices[idx_end_offset + 1 :]:
                row_end, col_end = positions[idx_end]
                row_diff = row_end - row_start
                col_diff = col_end - col_start
                steps_row = [
                    (row_diff // abs(row_diff), 0) for _ in range(1, abs(row_diff) + 1)
                ]
                steps_col = [
                    (0, col_diff // abs(col_diff)) for _ in range(1, abs(col_diff) + 1)
                ]
                start_with_row = (row_start + row_diff, col_start) in all_pos
                # Order of movement matters
                if start_with_row:
                    steps = steps_row + steps_col
                else:
                    steps = steps_col + steps_row
                set_shortest[idx_start][idx_end] = steps
                set_shortest[idx_end][idx_start] = [(-r, -c) for r, c in steps[::-1]]

    find_paths(pos_numeric, range(0, 11), numeric_shortest)
    # Key is where the button will move us, value is position of this key
    # Key A is (0, 0) on directional
    pos_directional = {}
    pos_directional[(-1, 0)] = (0, 1)
    pos_directional[(0, 0)] = (0, 2)
    pos_directional[(0, -1)] = (1, 0)
    pos_directional[(1, 0)] = (1, 1)
    pos_directional[(0, 1)] = (1, 2)
    dirs = [(0, 0), (1, 0), (-1, 0), (0, -1), (0, 1)]
    directional_shortest = {}
    for dir in dirs:
        directional_shortest[dir] = {}

    find_paths(pos_directional, dirs, directional_shortest)

    def steps_to_string(steps):
        ret = ""
        step_char = {}
        step_char[(0, 0)] = "A"
        step_char[(1, 0)] = "v"
        step_char[(-1, 0)] = "^"
        step_char[(0, 1)] = ">"
        step_char[(0, -1)] = "<"
        for step in steps:
            ret += step_char[step]
        return ret

    for code in codes:
        cur_val_num = 10
        cur_val_robot1 = (0, 0)
        cur_val_robot2 = (0, 0)
        all_steps = []
        for next_val_num in code:
            robot1_moves = []
            robot1_moves += numeric_shortest[cur_val_num][next_val_num]
            robot1_moves += [(0, 0)]
            for next_val_robot1 in robot1_moves:
                robot2_moves = []
                if next_val_robot1 in directional_shortest[cur_val_robot1]:
                    robot2_moves += directional_shortest[cur_val_robot1][
                        next_val_robot1
                    ]
                robot2_moves += [(0, 0)]
                for next_val_robot2 in robot2_moves:
                    man_moves = []
                    if next_val_robot2 in directional_shortest[cur_val_robot2]:
                        man_moves += directional_shortest[cur_val_robot2][
                            next_val_robot2
                        ]
                    man_moves += [(0, 0)]
                    all_steps += man_moves
                    cur_val_robot2 = next_val_robot2
                cur_val_robot1 = next_val_robot1
            cur_val_num = next_val_num
        print("".join(map(str, code)), steps_to_string(all_steps))
    for k, v in directional_shortest.items():
        print(k, v)


def day22_part1(fn: str):
    numbers = list(map(int, read_file_as_lines(fn)))

    def next(x: int, num_iter: int):
        prune_val: int = (1 << 24) - 1
        for _ in range(num_iter):
            x = (x * 64 ^ x) & prune_val
            x = (x // 32 ^ x) & prune_val
            x = (x * 2048 ^ x) & prune_val
        return x

    res = sum([next(x, 2000) for x in numbers])
    return res


def day22_part2(fn: str):
    numbers = list(map(int, read_file_as_lines(fn)))

    if fn.find("test") > -1:
        numbers = [1, 2, 3, 2024]

    def next(x: int, num_iter: int, valdiff: List[tuple]):
        prune_val: int = (1 << 24) - 1
        last_last = x % 10
        for _ in range(num_iter):
            x = (x * 64 ^ x) & prune_val
            x = (x // 32 ^ x) & prune_val
            x = (x * 2048 ^ x) & prune_val
            cur_last = x % 10
            valdiff.append((cur_last, cur_last - last_last))
            last_last = cur_last
        return x

    scores_total = {}
    for _, x in enumerate(numbers):
        valdiff = []
        _ = next(x, 2000, valdiff)
        scores = {}
        for idx_start in range(len(valdiff) - 3):
            combo = tuple([v for _, v in valdiff[idx_start : idx_start + 4]])
            if combo not in scores:
                score = valdiff[idx_start + 3][0]
                scores[combo] = score
                scores_total[combo] = scores_total.get(combo, 0) + score
    max_key = max(scores_total, key=scores_total.get)
    return scores_total[max_key]


def day23_part1(fn: str):
    def name_to_int(name: str):
        return ord(name[0]) * 1000 + ord(name[1])

    lines = read_file_as_lines(fn)
    pairs = list(map(lambda x: tuple(x.split("-")), lines))
    pairs = [(name_to_int(x), name_to_int(y)) for x, y in pairs]
    names = {}
    for v1, v2 in pairs:
        val_min = min(v1, v2)
        val_max = max(v1, v2)
        names[val_min] = names.get(val_min, []) + [val_max]
    combos = set()
    for v, k in names.items():
        k = sorted(k)
        val_first_char = [0] * 3
        val_first_char[0] = v // 1000
        for idx_low in range(len(k) - 1):
            val_low = k[idx_low]
            val_first_char[1] = val_low // 1000
            for idx_high in range(idx_low + 1, len(k)):
                val_high = k[idx_high]
                val_first_char[2] = val_high // 1000
                if val_low in names and val_high in names[val_low]:
                    if ord("t") in val_first_char:
                        combos.add((v, val_low, val_high))
    ret = len(combos)
    return ret


def day23_part2(fn: str):
    def name_to_int(name: str):
        return ord(name[0]) * 1000 + ord(name[1])

    lines = read_file_as_lines(fn)
    pairs = list(map(lambda x: tuple(x.split("-")), lines))
    pairs = [(name_to_int(x), name_to_int(y)) for x, y in pairs]
    pairs = [(min(x, y), max(x, y)) for x, y in pairs]
    pairs_sorted = sorted(pairs, key=lambda x: (x[0], x[1]))
    direct_con = {}
    for x, y in pairs_sorted:
        val_min = min(x, y)
        val_max = max(x, y)
        direct_con[val_min] = direct_con.get(val_min, []) + [val_max]

    groups = {}
    for x, y in pairs_sorted:
        val_min = min(x, y)
        val_max = max(x, y)
        do_add = True
        if val_min not in groups:
            do_add = True
        else:
            for val_cur in groups[val_min]:
                cur_min = min(val_cur, val_max)
                cur_max = max(val_cur, val_max)
                if cur_min not in direct_con:
                    continue
                if cur_max not in direct_con[cur_min]:
                    do_add = False
                    break
        if do_add:
            groups[val_min] = groups.get(val_min, []) + [val_max]
    max_group = max(groups.items(), key=lambda x: len(x[1]))
    vals = [max_group[0]] + max_group[1]
    names = [chr(x // 1000) + chr(x % 1000) for x in vals]
    return ",".join(names)


def day24_get_input(
    fn: str,
) -> tuple[List[tuple[str, int]],]:
    lines = read_file_as_lines(fn)
    idx_empty = lines.index("")
    lines_initial = lines[:idx_empty]
    lines_wires = lines[idx_empty + 1 :]
    initial = list(map(lambda line: line.split(": "), lines_initial))
    initial = [(x, int(y)) for x, y in initial]
    wires = list(map(lambda line: line.split(" -> "), lines_wires))
    wires = [tuple([y] + x.split(" ")) for x, y in wires]
    return initial, wires


def day24_get_z(
    initial: List[tuple[str, int]], wires: List[tuple[str, str, str, str]]
) -> int:
    vals = {}
    for name, val in initial:
        vals[name] = val
    zs = [var_out for var_out, _, _, _ in wires if var_out.startswith("z")]
    zs.sort()
    done = False
    while not done:
        for var_out, var_in_1, op, var_in_2 in wires:
            if var_out not in vals:
                if var_in_1 in vals and var_in_2 in vals:
                    val1 = vals[var_in_1]
                    val2 = vals[var_in_2]
                    val_out = 0
                    if op == "OR":
                        val_out = val1 | val2
                    if op == "XOR":
                        val_out = val1 ^ val2
                    if op == "AND":
                        val_out = val1 & val2
                    vals[var_out] = val_out
        cnt = 0
        for cur_var_z in zs:
            if cur_var_z in vals:
                cnt += 1
        if cnt == len(zs):
            done = True
    bits = [vals[x] for x in zs]
    res = 0
    for idx, bit in enumerate(bits):
        res += bit << idx
    return res


def day24_part1(fn: str):
    initial, wires = day24_get_input(fn)
    return day24_get_z(initial, wires)


def day24_part2(fn: str):
    if fn.find("test") > -1:
        return -1
    initial, wires = day24_get_input(fn)
    zs = [var_out for var_out, _, _, _ in wires if var_out.startswith("z")]
    num_z_vals = len(zs)
    x_bits = [v for k, v in initial[::-1] if k.startswith("x")]
    y_bits = [v for k, v in initial[::-1] if k.startswith("y")]
    x_bin_string = "".join([str(v) for v in x_bits])
    y_bin_string = "".join([str(v) for v in y_bits])
    x_val = int(x_bin_string, 2)
    y_val = int(y_bin_string, 2)
    z_exp_val = x_val + y_val
    z_exp_bin_string = f"{z_exp_val:0{num_z_vals}b}"
    z_act_val = day24_get_z(initial, wires)
    z_act_bin_string = f"{z_act_val:0{num_z_vals}b}"
    print(x_bin_string, x_val)
    print(y_bin_string, y_val)
    print(z_exp_bin_string, z_exp_val)
    print(z_act_bin_string, z_act_val)
    print(len(wires))
    all_inputs = {}
    for _, v1, _, v2 in wires:
        all_inputs[v1] = all_inputs.get(v1, 0) + 1
        all_inputs[v2] = all_inputs.get(v2, 0) + 1
    max_inputs = [k for k, v in all_inputs.items() if v == 2]
    min_inputs = [k for k, v in all_inputs.items() if v == 1]
    print(max_inputs)
    print(min_inputs)
    return 0
