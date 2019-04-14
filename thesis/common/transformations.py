import re


def string_to_list(str):
    return re.sub(r'^(\[\')|(\',\s\')|(\'])$', '|', str).split('|')[1:-1]
