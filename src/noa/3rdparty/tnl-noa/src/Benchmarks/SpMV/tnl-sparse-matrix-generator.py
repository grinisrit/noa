#!/usr/bin/python3

from scipy import sparse
from scipy.sparse import random
from scipy import stats
from scipy import io as sio
from numpy.random import default_rng

#sizes = [ 100, 500, 1000, 5000, 10000, 50000, 100000, 500000 ]
sizes = [ 100000 ]

#for size in sizes:
#   nnz_per_row = 1
#   while nnz_per_row < 50 and nnz_per_row <= size:
#      rng = default_rng()
#      rvs = stats.poisson(25, loc=10).rvs
#      S = random( size, size, density=nnz_per_row/size, random_state=rng, data_rvs=rvs)
#      print( f"Writing file matrix-{size}-{nnz_per_row}.mtx", end="\r" )
#      sio.mmwrite( f"matrix-{size}-{nnz_per_row}.mtx", S )
#      nnz_per_row = nnz_per_row + 1

for size in sizes:
   nnz_per_row = 1
   while nnz_per_row < 256 and nnz_per_row <= size:
      print( f"Creating matrix with {nnz_per_row} non-zeros per row")
      rows = []
      columns = []
      values = []
      row_idx = 0
      while row_idx < size:
         #print( f"Row idx = {row_idx}" )
         col_idx = max( 0, row_idx - (nnz_per_row - 1 ) / 2 )
         local_idx = 0
         while local_idx < nnz_per_row:
            rows.append( row_idx )
            columns.append( min( col_idx + local_idx, size - 1 ) )
            values.append( 1.0 )
            local_idx = local_idx + 1
         row_idx = row_idx + 1
      matrix = sparse.csr_matrix( ( values,(rows, columns)), shape=(size,size))
      print( f"Writing file matrix-{size}-{nnz_per_row}.mtx", end="\r" )
      sio.mmwrite( f"matrix-{size}-{nnz_per_row}.mtx", matrix )
      nnz_per_row = nnz_per_row + 1


