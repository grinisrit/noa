#!/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    if len(sys.argv) == 1:
        print(f"Usage: {sys.argv[0]} dump_file_1.txt dump_file_2.txt ...")
        exit(1)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection="3d")

    color = "bgrcmyk"

    oobCounter = 0
    for fi, fname in enumerate(sys.argv[1:]):
        with open(fname, "r") as dumpFile:
            x = []
            y = []
            z = []
            for line in dumpFile.readlines():
                if "particle " in line:
                    pnum = int(line.replace("particle ", ""))
                    if x:
                        if max(z) > 1e3: oobCounter += 1
                        ax.plot(x, y, z, color[fi], label=fname)
                        x = []
                        y = []
                        z = []
                    print(f"\r{pnum}", end="")
                    # if pnum > 1: break
                    continue
                p = line.strip().split(", ")
                x.append(float(p[0]))
                y.append(float(p[1]))
                z.append(float(p[2]))

    print("\nOK")
    print(f"{oobCounter} out of bounds")
    # legend = plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()
