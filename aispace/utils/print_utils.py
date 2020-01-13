# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-02 21:15
# @Author  : yingyuankai@aliyun.com
# @File    : print_utils.py

import logging
import collections
from pprint import pformat

__all__ = [
    "print_boxed",
    "print_aispace",
    "repr_ordered_dict"
]


def print_aispace(message, aispace_version):
    print('\n'.join(
        [
            '',
            '    ____     _  ______                                     ',
            r'   / /\ \   (_/  ___  |_______    _____     ______   ____  ',
            r'  / /--\ \  | \___   \   ____  \/  ___  \ /  _____| / ___\ ',
            r' / /----\ \ | |____\  \ |____/ |  /___\ |_| |_____| \_____ ',
            r'/_/      \_\|_|______/  |_____/ \_________\ ______\______/ ',
            '                      | |                                  ',
            '                      |_|                                  ',
            'AiSpace v{1} - {0}'.format(message, aispace_version),
            ''
        ]
    ))


def print_boxed(text, print_fun=print):
    box_width = len(text) + 2
    print_fun('')
    print_fun('╒{}╕'.format('═' * box_width))
    print_fun('│ {} │'.format(text.upper()))
    print_fun('╘{}╛'.format('═' * box_width))
    print_fun('')


def repr_ordered_dict(d):
    return '{\n  ' + ',\n  '.join('{}: {}'.format(x, pformat(y, indent=4))
                              for x, y in d.items()) + '\n}'


