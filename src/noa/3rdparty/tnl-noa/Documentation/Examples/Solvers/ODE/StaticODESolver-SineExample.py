#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np

###
## Enable latex for labels
plt.rcParams['text.usetex'] = True

###
# Parse the input file
f = open( sys.argv[1], 'r' )
x_lst = []
y_lst = []
for line in f:
    line = line.strip()
    a = line.split()
    x_lst.append( float( a[ 0 ] ) )
    y_lst.append( float( a[ 1 ] ) )

###
# Convert the data to NumPy array
x = np.array(x_lst)
y = np.array(y_lst)

###
# Draw the graph of u(t) using Matplotlib
fig, ax = plt.subplots()
ax.set_xlim( [0,10] )
ax.set_ylim( [-10,10] )
ax.plot(x, y, linewidth=2.0)
ax.set_xlabel( "$t$" )
ax.set_ylabel( "$u(t)$" )
plt.savefig( sys.argv[2] )
plt.close(fig)
