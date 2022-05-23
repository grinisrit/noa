#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D

plt.rcParams['text.usetex'] = True

###
# Enable latex for labels
f = open( sys.argv[1], 'r' )
current_sigma = 0.0
current_rho = 0.0
current_beta = 0.0
sigma_lst = []
rho_lst = []
beta_lst = []
x_lst = []
y_lst = []
z_lst = []
x_data = []
y_data = []
z_data = []
parameters = []
data = {}
size = 0
for line in f:
    line = line.strip()
    a = line.split()
    if not a:
        continue
    if a[ 0 ] == "#":
        if x_lst:
            parameters_tuple = ( current_sigma, current_rho, current_beta )
            parameters.append( parameters_tuple )
            data_tuple = ( np.array( x_lst ), np.array( y_lst ), np.array( z_lst ) )
            data[ parameters_tuple ] = data_tuple
            x_lst.clear()
            y_lst.clear()
            z_lst.clear()

        current_sigma = float( a[ 2 ] )
        current_rho = float( a[ 4 ] )
        current_beta = float( a [ 6 ] )
        if current_sigma not in sigma_lst:
            sigma_lst.append( current_sigma )
        if current_rho not in rho_lst:
            rho_lst.append( current_rho )
        if current_beta not in beta_lst:
            beta_lst.append( current_beta )
    else:
        x_lst.append( float( a[ 0 ] ) )
        y_lst.append( float( a[ 1 ] ) )
        z_lst.append( float( a[ 2 ] ) )
parameters_tuple = ( current_sigma, current_rho, current_beta )
parameters.append( parameters_tuple )
data_tuple = ( np.array( x_lst ), np.array( y_lst ), np.array( z_lst ) )
data[ parameters_tuple ] = data_tuple

###
# Draw the graph of u(t) using Matplotlib
sigma_n = len( sigma_lst )
sigma_idx = 1
for sigma in sigma_lst:
    rho_n   = len( rho_lst )
    beta_n  = len( beta_lst )

    fig, ax = plt.subplots( rho_n, beta_n, figsize=(8, 8), sharey=True, sharex=True )
    fig.suptitle( f'$\sigma={sigma}$')
    #ax = Axes3D(fig)  does not work with ax indexing
    rho_idx = 0
    beta_idx = 0
    for rho in rho_lst:
        for beta in beta_lst:
            parameters_tuple = ( sigma, rho, beta )
            data_tuple = data[ parameters_tuple ]
            ax[ rho_idx, beta_idx ].plot( data_tuple[ 1 ], data_tuple[ 2 ], linewidth=1.0 )
            if beta_idx == 0:
                ax[ rho_idx, beta_idx ].set_ylabel( f'$\\rho={rho}$' )
            if rho_idx == rho_n-1:
                ax[ rho_idx, beta_idx ].set_xlabel( f'$\\beta={beta}$' )
            beta_idx = beta_idx + 1
        beta_idx = 0
        rho_idx = rho_idx + 1

    plt.savefig( f"{sys.argv[2]}-{sigma_idx}.png" )
    sigma_idx = sigma_idx + 1
    plt.close(fig)




