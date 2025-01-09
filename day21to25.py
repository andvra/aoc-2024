from shared import *


def day21_part1(fn: str):
    if fn.find("real") > -1:
        return -1
    # Key A is number 10 (= idx 11) on numeric
    pos_numeric = [(3, 1)]
    for n in range(1, 10):
        row = 2 - (n - 1) // 3
        col = (n - 1) % 3
        pos_numeric.append((row, col))
    pos_numeric.append((3, 2))
    numeric_shortest = {}
    for idx in range(0, 12):
        numeric_shortest[idx] = {}
    for idx_start in range(0, 11):
        row_start, col_start = pos_numeric[idx_start]
        for idx_end in range(idx_start, 11):
            row_end, col_end = pos_numeric[idx_end]
            row_diff = row_end - row_start
            col_diff = col_end - col_start
            steps_row = [
                (row_diff // abs(row_diff), 0) for _ in range(1, abs(row_diff) + 1)
            ]
            steps_col = [
                (0, col_diff // abs(col_diff)) for _ in range(1, abs(col_diff) + 1)
            ]
            # Order of movement matters
            if row_diff > 0:
                steps = steps_col + steps_row
            else:
                steps = steps_row + steps_col
            numeric_shortest[idx_start][idx_end] = steps
            numeric_shortest[idx_end][idx_start] = [(-r, -c) for r, c in steps[::-1]]
    for k, v in numeric_shortest.items():
        print(k, v)
    # Key is where the button will move us, value is position of this key
    # Key A is (0, 0) on directional
    dir_placements = {}
    dir_placements[(-1, 0)] = (0, 1)
    dir_placements[(0, 0)] = (0, 2)
    dir_placements[(0, -1)] = (1, 0)
    dir_placements[(1, 0)] = (1, 1)
    dir_placements[(0, 1)] = (1, 2)
    dirs = [(0, 0), (1, 0), (-1, 0), (0, -1), (0, 1)]
    directional_shortest = {}
    for dir in dirs:
        directional_shortest[dir] = {}
    for idx_start in range(len(dir_placements)):
        dir_start = dirs[idx_start]
        row_start, col_start = dir_placements[dir_start]
        for idx_end in range(idx_start + 1, len(dir_placements)):
            dir_end = dirs[idx_end]
            row_end, col_end = dir_placements[dir_end]
            row_diff = row_end - row_start
            col_diff = col_end - col_start
            steps_row = [
                (row_diff // abs(row_diff), 0) for _ in range(1, abs(row_diff) + 1)
            ]
            steps_col = [
                (0, col_diff // abs(col_diff)) for _ in range(1, abs(col_diff) + 1)
            ]
            if row_diff > 0:
                steps = steps_row + steps_col
            else:
                steps = steps_col + steps_row
            directional_shortest[dir_start][dir_end] = steps
            directional_shortest[dir_end][dir_start] = [
                (-r, -c) for r, c in steps[::-1]
            ]
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
    if fn.find("real") > -1:
        return -1
    lines = read_file_as_lines(fn)
    pairs = list(map(lambda x: tuple(x.split("-")), lines))
    names = {}
    lans = []
    for pc1, pc2 in pairs:
        new_id = -1
        cur_lan_ids = set()
        if pc1 in names:
            cur_lan_ids.add(names[pc1])
        if pc2 in names:
            cur_lan_ids.add(names[pc2])
        if len(cur_lan_ids) == 2:
            lan_id_min = min(cur_lan_ids)
            lan_id_max = max(cur_lan_ids)
            lans[lan_id_min].update(lans[lan_id_max])
            lans[lan_id_max].clear()
            new_id = lan_id_min
        if len(cur_lan_ids) == 1:
            new_id = cur_lan_ids.pop()
        if new_id == -1:
            lans.append(set())
            new_id = len(lans) - 1
        names[pc1] = new_id
        names[pc2] = new_id
        lans[new_id].add(pc1)
        lans[new_id].add(pc2)
    print(names)
    print(lans)
    return 0
