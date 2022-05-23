#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D

plt.rcParams['text.usetex'] = True

f = open( sys.argv[1], 'r' )
current_time = 0.0
time_lst = []
x_lst = []
u_lst = []
x_data = []
u_data = []
size = 0
for line in f:
    line = line.strip()
    a = line.split()
    if not a:
        continue
    if a[ 0 ] == "#":
        if u_lst:
            time_lst.append( current_time )
            u_data.append( np.array( u_lst ) )
            if not x_data:
                x_data.append( np.array( x_lst ) )
            x_lst.clear()
            u_lst.clear()

        current_time = float( a[ 3 ] )
        time_lst.append( current_time )
    else:
        x_lst.append( float( a[ 0 ] ) )
        u_lst.append( float( a[ 1 ] ) )
if u_lst:
    time_lst.append( current_time )
    u_data.append( np.array( u_lst ) )
    if not x_data:
        x_data.append( np.array( x_lst ) )
    x_lst.clear()
    u_lst.clear()


fig, ax = plt.subplots( 1, 1, figsize=(4, 4) )
for u in u_data:
    ax.plot( x_data[ 0 ], u, linewidth=2.0 )

ax.set_ylabel( f'$ u(x) $' )
ax.set_xlabel( f'$x $' )
plt.savefig( f"{sys.argv[2]}.png" )
plt.close(fig)
