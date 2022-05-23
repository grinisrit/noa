#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np

###
# Enable latex for labels
plt.rcParams['text.usetex'] = True

###
# Parse the input file
f = open( sys.argv[1], 'r' )
current_c = 0.0
x_lst = []
u_lst = []
x_data = []
u_data = []
parameters = []

for line in f:
    line = line.strip()
    a = line.split()
    if not a:
        continue
    if a[ 0 ] == "#":
        if x_lst:
            parameters.append( current_c )
            u_data.append( np.array( u_lst ) )
            if not x_data:
                x_data.append( np.array( x_lst ) )
            u_lst.clear()

        current_c = float( a [ 3 ] )
    else:
        if not x_data:
            x_lst.append( float( a[ 0 ] ) )
        u_lst.append( float( a[ 1 ] ) )
parameters.append( current_c )
u_data.append( np.array( u_lst ) )

###
# Draw the graph of u(t) using Matplotlib
n = len( parameters )
fig, ax = plt.subplots( 1, n, figsize=(15, 3), sharey=True )
idx = 0
for u in u_data:
    ax[ idx ].plot( x_data[0], u, linewidth=2.0, label=f"$c={parameters[idx]}$" )
    ax[ idx ].set_xlabel( "t" )
    ax[ idx ].set_ylabel( "u(t)" )
    idx = idx + 1
plt.savefig( sys.argv[2] )
plt.close(fig)




