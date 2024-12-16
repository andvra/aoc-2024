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
    return 0


def day6_part2(fn):
    return 0


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


def aoc_2024():
    for num_day in range(1, 26):
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
            res_real = func(fn_real)
            print(f"Day {num_day} (part {part}): {res_test} / {res_real}")


if __name__ == "__main__":
    aoc_2024()
