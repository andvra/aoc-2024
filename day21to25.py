from shared import *


def day21_part1(fn: str):
    if fn.find("real") > -1:
        return -1

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
    # TODO: Now we now how to move on the numeric keypad. Do the same for directional keypad
