# -*- coding: utf-8 -*-
import os
import sys


def write_helpfile(obj):
    """
    Function to write help(obj) to file in current working directory.
    """
    filename = obj.__class__.__name__ + '_help.txt'
    try:
        with open(filename, 'w') as f:
            t = sys.stdout
            sys.stdout = f
            help(obj)
            sys.stdout = t
        if not os.path.isfile(filename):
            raise Exception('No file written')
    except Exception as e:
        print('Error in helpfilewriter: ', e)
    else:
        print(f'Class {obj.__class__.__name__} help file {filename} saved to working directory.')
