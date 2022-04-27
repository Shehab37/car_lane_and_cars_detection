from parameters import *
from functions import *
from classes import *


if __name__ == "__main__":

    # argv = ['main.py', input_path, , 'debug']

    input_path = sys.argv[1]

    debug = (sys.argv[2] == '1')
    if debug:
        print('debug mode')

    create_output(input_path, debug=debug)
