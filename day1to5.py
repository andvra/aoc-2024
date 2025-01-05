import re
from shared import *


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
    start_indices = [m.start() for m in re.finditer(r"mul\(", line)]
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
