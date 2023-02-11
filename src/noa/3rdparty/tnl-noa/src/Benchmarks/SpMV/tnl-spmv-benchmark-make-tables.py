#! /usr/bin/env python3

import os
import re
import math
import pandas

from collections import defaultdict
from TNL.LogParser import LogParser

"""
Sparse matrix formats as they appear in the log file.
"""
cpu_matrix_formats = [ 'CSR',
                       'Ellpack', 'Ellpack Legacy',
                       'SlicedEllpack', 'SlicedEllpack Legacy',
                       'ChunkedEllpack', 'ChunkedEllpack Legacy',
                       'BiEllpack', 'BiEllpack Legacy' ]

gpu_matrix_formats = [ 'CSR Legacy Scalar', 'CSR Legacy Vector', 'CSR Legacy MultiVector',
                       'CSR Legacy Light', 'CSR Legacy Light2', 'CSR Legacy Light3', 'CSR Legacy Light4', 'CSR Legacy Light5', 'CSR Legacy Light6', 'CSR Legacy LightWithoutAtomic',
                       'CSR Legacy Adaptive',
                       'CSR< Scalar >', 'CSR< Vector >', 'CSR< Hybrid >', 'CSR< Adaptive >',
                       'Ellpack', 'Ellpack Legacy',
                       'SlicedEllpack', 'SlicedEllpack Legacy',
                       'ChunkedEllpack', 'ChunkedEllpack Legacy',
                       'BiEllpack', 'BiEllpack Legacy' ]
"""
CPU formats to be compared
"""
cpu_comparison_formats = { 'CSR' : 'CSR Legacy Scalar',
                           'Ellpack' : 'Ellpack Legacy',
                           'SlicedEllpack' : 'SlicedEllpack Legacy',
                           'BiEllpack' : 'BiEllpack Legacy'
                          }

"""
GPU formats to be compared
"""
gpu_comparison_formats = { 'CSR< Scalar >' : 'CSR Legacy Scalar',
                           'CSR< Vector >' : 'CSR Legacy Vector',
                           'CSR< Hybrid >' : 'CSR Legacy LightWithoutAtomic',
                           'CSR< Adaptive >' : 'CSR Legacy Adaptive',
                           'Ellpack' : 'Ellpack Legacy',
                           'SlicedEllpack' : 'SlicedEllpack Legacy',
                           'BiEllpack' : 'BiEllpack Legacy'
                          }
#pandas.options.display.float_format = "{:.2f}".format
pandas.options.display.float_format = "{:.2e}".format
pandas.options.display.width = 0    # auto-detect terminal width for formatting
pandas.options.display.max_rows = None

def slugify(s):
   s = str(s).strip().replace(' ', '_')
   return re.sub(r'(?u)[^-\w.]', '', s)


def parse_file(fname):
    parser = LogParser()
    for metadata, df in parser.readFile(fname):
        yield df

def calculate_efficiency(df, nodes_col_index, base_column=None):
    if base_column is None:
        base_column = df[df.columns[0]]
    eff_rows = []
    for i in df.index:
        row = df.loc[i]
        eff_row = row.copy()
        eff_idx = ("eff", *row.name[1:])
        base = base_column[i]
        for j in row.index:
            if isinstance(j, int):
                n = j
            else:
                n = j[nodes_col_index]
            eff_row[j] = base / row[j] / n
        eff_rows.append(eff_row)
    eff_df = pandas.DataFrame(eff_rows)
    eff_df.index = pandas.MultiIndex.from_tuples(eff_df.index)
    eff_df = eff_df.rename(index={"time": "eff"})
    return df.append(eff_df)

log_files = ["sparse-matrix-benchmark.log"]
print( "Parsing log file..." )

dfs = []
for f in log_files:
    for df in parse_file(f):
        dfs.append(df)

df = pandas.concat(dfs)

## Post-processing
print( "Postprocessing data frame..." )
# Drop norms of results differences
#df.drop(columns=['CSR Diff.Max','CSR Diff.L2'], axis=1, level=1, inplace=True )

# show matrix formats as columns
df = df.unstack()
df = df.reorder_levels([2, 0, 1], axis=1)
df.sort_index(axis=1, inplace=True)

# Drop CPU speedup
for cpu_format in cpu_matrix_formats:
   df.drop(columns=( cpu_format, 'CPU','speedup'), axis=1, inplace=True )

#print( "Exporting data frame to log.html..." )
#pandas.options.display.float_format = '{:,.4f}'.format
#df.to_html("log.html")

print( "Computing speed-up of formats...")
# Add speedup compared to CSR and cuSparse
for cpu_format in cpu_matrix_formats:
   if cpu_format != 'CSR':
      df[cpu_format, "CPU", "CSR speedup"] = df[cpu_format, "CPU", "time"] / df["CSR","CPU", "time"]

for gpu_format in gpu_matrix_formats:
   df[ gpu_format, "GPU", "cuSparse speedup"] = df[ gpu_format,"GPU", "time"] / df["cuSparse", "GPU", "time"]

# Add speedup compared to legacy formats
for format in cpu_comparison_formats:
   other_format = cpu_comparison_formats[ format ]
   df[ format, "CPU", f"{other_format} speedup"]  = df[ format, "CPU", "time"] / df[ other_format,  "CPU", "time"]

for format in gpu_comparison_formats:
   other_format = gpu_comparison_formats[ format ]
   df[ format, "GPU", f"{other_format} speedup"]  = df[ format, "GPU", "time"] / df[ other_format,  "GPU", "time"]

print( "Exporting data frame to log.html..." )
pandas.options.display.float_format = '{:,.4f}'.format
df.to_html("log.html")

"""
Extract columns of reference formats on GPU
"""
print( "Preparing data for graph analysis..." )
df['cuSparse-bandwidth'                        ] = df[ 'cuSparse','GPU','bandwidth']
for gpu_format in gpu_matrix_formats:
   df[ gpu_format + ' Bandwidth' ] = df[ gpu_format,'GPU','bandwidth']

"""
Sort by cuSparse
"""
df.sort_values(by=["cuSparse-bandwidth"],inplace=True,ascending=False)
cuSparse_list = df['cuSparse-bandwidth'].tolist()
cusparse_comparison = defaultdict( list )
for gpu_format in gpu_matrix_formats:
   cusparse_comparison[ gpu_format ] = df[ gpu_format, "GPU", "bandwidth" ].tolist()

"""
Sort by comparison formats
"""
formats_comparison = defaultdict( list )
for format in gpu_comparison_formats:
   df.sort_values(by=[f"{format} Bandwidth"],inplace=True,ascending=False)
   formats_comparison[ format ] = df[format, "GPU", "bandwidth"].tolist()
   formats_comparison[ gpu_comparison_formats[ format ] ] = df[gpu_comparison_formats[ format ], "GPU", "bandwidth"].tolist()

"""
Writting gnuplot source files
"""
print( "Writing gnuplot files..." )

for gpu_format in gpu_matrix_formats:
   filename = "cusparse-" + slugify( gpu_format ) + ".gplt"
   data = cusparse_comparison[ gpu_format ]
   out_file = open( filename, "w" )
   i = 0
   for x in cuSparse_list:
      if str( x ) != "nan":
         if ( str(cusparse_comparison[ gpu_format ][ i ] ) != "nan" ):
            out_file.write( f"{i+1} {x} {data[ i ]} \n" )
            i = i + 1;
   out_file.close()

for format in gpu_comparison_formats:
   out_file = open( f"{slugify(format)}-gpu-comparison.gplt", "w" )
   data = formats_comparison[ format ]
   other_data = formats_comparison[ gpu_comparison_formats[ format ] ]
   i = 0
   for x in data:
      if str( x ) != "nan":
         if str( other_data[ i ] ) != "nan":
            out_file.write( f"{i+1} {x} {other_data[ i ]}\n" )
      i = i + 1
   out_file.close()

"""
Generating gnuplot script
"""
print( "Generating Gnuplot script..." )

gnuplot_file = open( "gnuplot.gplt", "w" )
gnuplot_file.write( r"""
set terminal postscript lw 3 20 color
set grid
set xlabel 'Matrix'
set xtics 250
set ylabel 'Bandwidth GB/sec'
""" )
for gpu_format in gpu_matrix_formats:
   filename = "cusparse-" + slugify( gpu_format ) + ".gplt"
   gnuplot_file.write( f"set output 'cusparse-vs-{slugify(gpu_format)}.eps' \n" )
   gnuplot_file.write( f"plot '{filename}' using 1:2 title '' with dots linewidth 2 lt rgb 'red', " )
   gnuplot_file.write( f" '{filename}' using 1:2 title 'cuSparse' with lines linewidth 0.5 lt rgb 'red', " )
   gnuplot_file.write( f" '{filename}' using 1:3 title '' with dots linewidth 2 lt rgb 'green', " )
   gnuplot_file.write( f" '{filename}' using 1:3 title '{gpu_format}' with lines linewidth 0.5 lt rgb 'green'  \n" )


for format in gpu_comparison_formats:
   filename = f"{slugify(format)}-gpu-comparison.gplt"
   data = formats_comparison[ format ]
   other_data = formats_comparison[ gpu_comparison_formats[ format ] ]
   gnuplot_file.write( f"set output '{slugify(format)}-vs-{slugify(gpu_comparison_formats[ format ])}.eps' \n" )
   gnuplot_file.write( f"plot '{filename}' using 1:2 title '' with dots linewidth 2 lt rgb 'red', " )
   gnuplot_file.write( f" '{filename}' using 1:2 title '{format}' with lines linewidth 0.5 lt rgb 'red'," )
   gnuplot_file.write( f" '{filename}' using 1:3 title '' with dots linewidth 2 lt rgb 'blue', " )
   gnuplot_file.write( f" '{filename}' using 1:3 title '{gpu_comparison_formats[ format ]}' with lines linewidth 0.5 lt rgb 'blue' \n" )

gnuplot_file.close()

"""
Executing Gnuplot
"""

print( "Executing Gnuplot ..." )
os.system( "gnuplot gnuplot.gplt" )

"""
Converting files to PDF
"""
print( "Converting files to PDF ..." )
for gpu_format in gpu_matrix_formats:
   filename = "cusparse-vs-" + slugify( gpu_format ) + ".eps"
   os.system( f"epstopdf --autorotate All {filename}" )

for format in gpu_comparison_formats:
   filename = slugify(format) + "-vs-" + slugify(gpu_comparison_formats[ format ]) + ".eps"
   os.system( f"epstopdf --autorotate All {filename}" )

"""
Deleting temporary files
"""
print( "Deleting temprary files..." )
for gpu_format in gpu_matrix_formats:
   filename = "cusparse-" + slugify( gpu_format ) + ".gplt"
   os.system( f"rm {filename}" )
   filename = "cusparse-vs-" + slugify( gpu_format ) + ".eps"
   os.system( f"rm {filename}" )

for format in gpu_comparison_formats:
   filename = f"{slugify(format)}-gpu-comparison.gplt"
   os.system( f"rm {filename}" )
   filename = slugify(format) + "-vs-" + slugify(gpu_comparison_formats[ format ]) + ".eps"
   os.system( f"rm {filename}" )
os.system( "rm gnuplot.gplt" )
