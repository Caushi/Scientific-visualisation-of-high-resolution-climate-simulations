"""
Description:    visualisation tool for .nc files for climate simulations
Author:         Claudio Cannizzaro
Created:        18.03.2022

"""

import sys
from Visualization import Visualization


def main():
    if (len(sys.argv)) < 2:
        print("You need to specify a path to the json-Parm-file")
        return 1
    args = sys.argv[1]
    vis = Visualization(args)

    vis.visualize()


if __name__ == "__main__":
    main()
