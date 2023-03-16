#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math
from os.path import exists

devices = [ "sequential", "host", 'cuda' ]
precisions = [ "float", "double" ]
dims = [ "1D", "2D", "3D" ]

#####
# Parse input files
parsed_lines = []
for device in devices:
    for precision in precisions:
        for dim in dims:
            filename = f"tnl-benchmark-grid-{dim}-{device}-{precision}.json"
            if not exists( filename ):
                print( f"Skipping non-existing input file {filename} ...." )
                continue
            print( f"Parsing input file {filename} ...." )
            with open( filename ) as f:
                lines = f.readlines()
                for line in lines:
                    parsed_line = json.loads(line)
                    parsed_lines.append( parsed_line )

df = pd.DataFrame(parsed_lines)
keys = ['time', 'bandwidth' ]

for key in keys:
   if key in df.keys():
      df[key] = pd.to_numeric(df[key])

df.to_html( f'tnl-benchmark-grid.html' )
