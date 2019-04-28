import re


def swap_series(full_table, s1, s2):  # swaps two series
    x = full_table[s1]
    full_table = full_table.drop(s1, axis=1)
    full_table[s1] = full_table[s2]
    full_table = full_table.drop(s2, axis=1)
    full_table[s2] = x
    return full_table


def switch_series(full_table, s1, s2):  # swaps s1, s2, deletes s2
    full_table = full_table.drop(s1, axis=1)
    full_table[s1] = full_table[s2]
    full_table = full_table.drop(s2, axis=1)
    return full_table


def string_to_list(str):
    return re.sub(r'^(\[\')|(\',\s\')|(\'])$', '|', str).split('|')[1:-1]
