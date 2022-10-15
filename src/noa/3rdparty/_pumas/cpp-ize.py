#!/bin/env python3

import os
import re

print("Turning PUMAS to C++...")

# Replace 'catch' keyword
os.system("sed -i 's/\([^a-z^A-Z^0-9^_]\)\(catch\)\([^a-z^A-Z^0-9^_]\)/\\1_\\2\\3/g' pumas.c")

# Receive other compiler errors
os.system("g++ -c pumas.c -fpermissive -w 2> errors")

with open("pumas.c", "r") as pumas:
    with open("errors", "r") as errors:
        lines = errors.readlines()

        lineIdx = 0
        gotoName = None
        gotoFrom = None
        gotoTo = None

        gotos = {}
        structs = []

        for lineIdx in range(len(lines)):
            line = lines[lineIdx]

            if not (gotoName is None):
                if f"goto {gotoName};" in line:
                    rs = re.search(r"([0-9]+).*goto", line)
                    gotoFrom = rs.group(1)

                    if (gotoName, gotoTo) in gotos:
                        gotos[(gotoName, gotoTo)].append(gotoFrom)
                    else:
                        gotos[(gotoName, gotoTo)] = [ gotoFrom ]
                    gotoName = None
                    gotoFrom = None
                    gotoTo = None
                    continue

            if "error: jump to label" in line:
                rs = re.search(r"pumas.c:([0-9]+):.*jump to label .([a-zA-Z0-9_]+).", line)
                gotoTo = rs.group(1)
                gotoName = rs.group(2)
            elif "error: designator order for field" in line:
                rs = re.search(r"pumas.c:([0-9]+):.*designator order for field .([a-zA-Z0-9_]+).", line)
                structs.append(rs.group(1))
            elif "error: " in line:
                print(line)

        # Divide the code into blocks
        blocks = []
        for goto in gotos:
            froms = gotos[goto]
            for _from in froms:
                blocks.append([ int(_from), int(goto[1]) ])

        def intersect(b1, b2):
            if b1[0] > b2[0]:
                return intersect(b2, b1)
            # b1[0] <= b2[0]
            if b1[1] > b2[0]:
                if b1[1] < b2[1]:
                    return [ [b1[0], b2[0]], [b2[0], b1[1]], [b1[1], b2[1]] ]
                else:
                    # b1 includes b2
                    return None
            # Blocks do not intersect
            return None

        # Remove block intersections
        b1Idx = 0
        maxb1 = 0
        while b1Idx < len(blocks):
            if b1Idx > maxb1: maxb1 = b1Idx
            b2Idx = b1Idx + 1
            while b2Idx < len(blocks):
                print("\rBlocks {:5d} {:5d} | {:5d} total - {}%".format(b1Idx, b2Idx, len(blocks), int(maxb1 * 100 / len(blocks))), end="")
                inter = intersect(blocks[b1Idx], blocks[b2Idx])
                if inter is None:
                    b2Idx += 1
                    continue
                blocks.pop(b2Idx)
                blocks.pop(b1Idx)
                for newBlock in inter:
                    blocks.append(newBlock)
                b1Idx = -1
                break
            b1Idx += 1

        print("")
        print("Done")

        pumasLines = pumas.readlines()

        # Remove C-style struct initializers
        for struct in structs:
            lIdx = int(struct) - 1
            pumasLines[lIdx] = "";
            # Search for declaration
            name = None
            nameLine = None
            while True:
                rs = re.search(r"\s*struct ([a-zA-Z_0-9]+) ([a-zA-Z_0-9]+) = {", pumasLines[lIdx])
                if rs is None:
                    lIdx -= 1
                    continue

                name = rs.group(2)
                pumasLines[lIdx] = f"struct {rs.group(1)} {name} = {{}};\n"
                nameLine = lIdx
                break

            lIdx = int(struct) - 2
            while lIdx > nameLine:
                pLine = pumasLines[lIdx]
                rs = re.search(r"\s*.([a-zA-Z0-9_]+) = (.+)", pLine)
                val = rs.group(2)
                if val[-1] == ",":
                    val = val[0:-1]
                pumasLines[lIdx] = f"{name}.{rs.group(1)} = {val};\n"
                lIdx -= 1

        # Add isolating blocks
        for blockIdx, block in enumerate(blocks):
            rs = re.search(r"\s*}\s*", pumasLines[block[0]])
            if rs is None:
                pumasLines.insert(block[0], "{\n")
            elif rs.group(0) == pumasLines[block[0]]:
                pumasLines.insert(block[0] + 1, "{\n")
            else:
                pumasLines.insert(block[0], "{\n")
            pumasLines.insert(block[1], "}\n")
            for b2Idx in range(blockIdx + 1, len(blocks)):
                b2 = blocks[b2Idx]
                if b2[0] > block[0]:
                    b2[0] += 1
                if b2[0] > block[1]:
                    b2[0] += 1
                if b2[1] > block[0]:
                    b2[1] += 1
                if b2[1] > block[1]:
                    b2[1] += 1

        with open("pumas_new.c", "w+") as pumasWrite:
            pumasWrite.write(
                    """/*
* This file differs from original pumas.c
* It is generated by cpp-ize.py script from https://github.com/grinisrit/noa
* The changes are aimed at making it compile with g++ with -fpermissive flag
*/
""")
            for pLine in pumasLines:
                pumasWrite.write(pLine)

os.system("mv pumas_new.c pumas.c")
os.system("rm errors")

