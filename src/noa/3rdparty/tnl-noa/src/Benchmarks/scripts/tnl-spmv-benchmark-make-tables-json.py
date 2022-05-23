#!/usr/bin/python3

import os
import json
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import numpy as np
import math

#Latex fonst set-up

#plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "sans-serif",
#   "font.sans-serif": ["Helvetica"]})
#
# for Palatino and other serif fonts use:
#plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "serif",
#   "font.serif": ["Palatino"],
#})


####
# A map of rgb points in your distribution
# [distance, (r, g, b)]
# distance is percentage from left edge
# https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python/50784012#50784012
heatmap = [
    [0.0,  (0.1, 0.1, 1.0)],
 #  [0.20, (0, 0, .5)],
 #  [0.40, (0, .5, 0)],
    [0.40, (0.1, 1.0, 0.1)],
#   [0.80, (.75, .75, 0)],
#   [0.90, (1.0, .75, 0)],
    [1.00, (1.0, 0.1, 0.1)],
]

def gaussian(x, a, b, c, d=0):
    return a * math.exp(-(x - b)**2 / (2 * c**2)) + d

def color_map(x, width=100, map=[], spread=1):
    width = float(width)
    r = sum([gaussian(x, p[1][0], p[0] * width, width/(spread*len(map))) for p in map])
    g = sum([gaussian(x, p[1][1], p[0] * width, width/(spread*len(map))) for p in map])
    b = sum([gaussian(x, p[1][2], p[0] * width, width/(spread*len(map))) for p in map])
    return min(1.0, r), min(1.0, g), min(1.0, b)

#for x in range(im.size[0]):
#    r, g, b = pixel(x, width=im.size[0], map=heatmap)
#    r, g, b = [int(256*v) for v in (r, g, b)]
#    for y in range(im.size[1]):
#        ld[x, y] = r, g, b


####
# Helper function
def slugify(s):
   s = str(s).strip().replace(' ', '_')
   return re.sub(r'(?u)[^-\w.]', '', s)

def latexFormatName( name ):
   name = name.replace('<','')
   name = name.replace('>','')
   name = name.replace( 'Light  Automatic ', '')
   #print( f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{name}~~~')
   if name == 'CSR':
      return 'CSR on CPU'
   if name == 'cusparse':
      return 'cuSPARSE'
   if 'SlicedEllpack' in name:
      return name.replace( 'SlicedEllpack', 'Sliced Ellpack' )
   if 'ChunkedEllpack' in name:
      return name.replace( 'ChunkedEllpack', 'Chunked Ellpack' )
   if 'BiEllpack' in name:
      return name.replace( 'BiEllpack', 'Bisection Ellpack' )
   if 'CSR Scalar' in name:
      return name.replace( 'CSR Scalar', 'Scalar CSR' )
   if 'CSR Vector' in name:
      return name.replace( 'CSR Vector', 'Vector CSR' )
   if 'CSR Light' in name:
      return name.replace( 'CSR Light', 'Light CSR' )
   if 'CSR Adaptive' in name:
      return name.replace( 'CSR Adaptive', 'Adaptive CSR' )
   return name

####
# Create multiindex for columns
def get_multiindex( input_df, formats ):
   level1 = [ 'Matrix name', 'rows', 'columns' ]
   level2 = [ '',            '',     ''        ]
   level3 = [ '',            '',     ''        ]
   level4 = [ '',            '',     ''        ]
   df_data = [[ ' ',' ',' ']]
   for format in formats:
      for device in ['CPU','GPU']:
         for data in ['bandwidth', 'time', 'diff.max' ]: #,'time','speed-up','non-zeros','stddev','stddev/time','diff.max','diff.l2']:
            level1.append( format )
            level2.append( device )
            level3.append( data )
            level4.append( '' )
            df_data[ 0 ].append( ' ' )
      if not format in [ 'cusparse', 'CSR' ]:
         for speedup in [ 'cusparse', 'CSR CPU']:
            level1.append( format )
            level2.append( 'GPU' )
            level3.append( 'speed-up')
            level4.append( speedup )
            df_data[ 0 ].append( ' ' )
      if 'Binary' in format:
         level1.append( format )
         level2.append( 'GPU' )
         level3.append( 'speed-up')
         level4.append( 'non-binary' )
         df_data[ 0 ].append( ' ' )
      if 'Symmetric' in format:
         level1.append( format )
         level2.append( 'GPU' )
         level3.append( 'speed-up')
         level4.append( 'non-symmetric' )
         df_data[ 0 ].append( ' ' )
      if format == 'CSR< Light > Automatic' or format == 'CSR< Light > Automatic Light':
         level1.append( format )
         level2.append( 'GPU' )
         level3.append( 'speed-up')
         level4.append( 'LightSpMV Vector' )
         df_data[ 0 ].append( ' ' )
      if format == 'TNL Best':
         level1.append( format )
         level2.append( 'GPU' )
         level3.append( 'format')
         level4.append( '' )
         df_data[ 0 ].append( ' ' )

   multiColumns = pd.MultiIndex.from_arrays([ level1, level2, level3, level4 ] )
   return multiColumns, df_data

####
# Convert input table to better structured one
def convert_data_frame( input_df, multicolumns, df_data, begin_idx = 0, end_idx = -1 ):
   frames = []
   in_idx = 0
   out_idx = 0
   #max_out_idx = max_rows
   if end_idx == -1:
      end_idx = len(input_df.index)
   best_count = 0
   while in_idx < len(input_df.index) and out_idx < end_idx:
      matrixName = input_df.iloc[in_idx]['matrix name']
      df_matrix = input_df.loc[input_df['matrix name'] == matrixName]
      if out_idx >= begin_idx:
         print( f'{out_idx} : {in_idx} / {len(input_df.index)} : {matrixName}' )
      else:
         print( f'{out_idx} : {in_idx} / {len(input_df.index)} : {matrixName} - SKIP' )
      aux_df = pd.DataFrame( df_data, columns = multicolumns, index = [out_idx] )
      best_bw = 0
      for index,row in df_matrix.iterrows():
         aux_df.iloc[0]['Matrix name'] = row['matrix name']
         aux_df.iloc[0]['rows']        = row['rows']
         aux_df.iloc[0]['columns']     = row['columns']
         current_format = row['format']
         current_device = row['device']
         #print( current_format + " / " + current_device )
         bw = pd.to_numeric(row['bandwidth'], errors='coerce')
         time = pd.to_numeric(row['time'], errors='coerce')
         diff_max = pd.to_numeric(row['CSR Diff.Max'], errors='coerce')
         aux_df.iloc[0][(current_format,current_device,'bandwidth','')] = bw
         aux_df.iloc[0][(current_format,current_device,'time','')] = time
         aux_df.iloc[0][(current_format,current_device,'diff.max','')] = diff_max
         if( current_device == 'GPU' and
             not 'Binary' in current_format and
             not 'Symmetric' in current_format and
             not 'Legacy' in current_format and
             not 'cusparse' in current_format and
             not 'LightSpMV' in current_format and
             not 'Hybrid' in current_format and
             current_format != 'CSR< Light > Automatic' and
             bw > best_bw ):
            best_bw = bw
            best_format = current_format
         if current_format == 'cusparse':
            cusparse_bw = bw
         #aux_df.iloc[0][(current_format,current_device,'time')]        = row['time']
         #aux_df.iloc[0][(current_format,current_device,'speed-up')]    = row['speedup']
         #aux_df.iloc[0][(current_format,current_device,'non-zeros')]   = row['non-zeros']
         #aux_df.iloc[0][(current_format,current_device,'stddev')]      = row['stddev']
         #aux_df.iloc[0][(current_format,current_device,'stddev/time')] = row['stddev/time']
         #aux_df.iloc[0][(current_format,current_device,'diff.max')]    = row['CSR Diff.Max']
         #aux_df.iloc[0][(current_format,current_device,'diff.l2')]    = row['CSR Diff.L2']
      aux_df.iloc[0][('TNL Best','GPU','bandwidth','')] = best_bw
      if best_bw > cusparse_bw:
         aux_df.iloc[0][('TNL Best','GPU','format','')] = best_format
      else:
         aux_df.iloc[0][('TNL Best','GPU','format','')] = 'cusparse'
      best_count += 1
      if out_idx >= begin_idx:
         frames.append( aux_df )
      out_idx = out_idx + 1
      in_idx = in_idx + len(df_matrix.index)
   result = pd.concat( frames )
   return result

####
# Compute speed-up of particular formats compared to Cusparse on GPU and CSR on CPU
def compute_cusparse_speedup( df, formats ):
   for device in [ 'CPU', 'GPU' ]:
      for format in formats:
         if not format in [ 'cusparse', 'CSR' ]:
            print( 'Adding speed-up for ', format )
            try:
               format_times_list = df[(format,device,'time')]
            except:
               continue
            cusparse_times_list = df[('cusparse','GPU','time')]
            csr_times_list = df[('CSR','CPU','time')]
            cusparse_speedup_list = []
            csr_speedup_list = []
            for( format_time, cusparse_time, csr_time ) in zip( format_times_list, cusparse_times_list,csr_times_list ):
               if( device == 'GPU' ):
                  try:
                     cusparse_speedup_list.append( cusparse_time / format_time  )
                  except:
                     cusparse_speedup_list.append(float('nan'))
               try:
                  csr_speedup_list.append( csr_time / format_time  )
               except:
                  csr_speedup_list.append(float('nan'))
            if( device == 'GPU' ):
               df[(format,'GPU','speed-up','cusparse')] = cusparse_speedup_list
            df[(format,device,'speed-up','CSR CPU')] = csr_speedup_list

####
# Compute speedup of Light CSR
def compute_csr_light_speedup( df, formats ):
   for light in [ 'CSR< Light > Automatic', 'CSR< Light > Automatic Light']:
      if light in formats:
         csr_light_bdw_list = df[(light,'GPU','bandwidth')]
         light_spmv_bdw_list = df[('LightSpMV Vector','GPU','bandwidth')]

         csr_light_speedup_list = []
         for ( csr_light_bdw, light_spmv_bdw ) in zip(csr_light_bdw_list,light_spmv_bdw_list):
            try:
               csr_light_speedup_list.append( csr_light_bdw / light_spmv_bdw  )
            except:
               csr_light_speedup_list.append(float('nan'))
         df[(light,'GPU','speed-up','LightSpMV Vector')] = csr_light_speedup_list

####
# Compute speed-up of binary formats
def compute_binary_speedup( df, formats ):
   for format in formats:
      if 'Binary' in format:
         non_binary_format = format.replace( 'Binary ', '' )
         print( f'Adding speed-up of {format} vs {non_binary_format}' )
         format_bdw_list = df[(format,'GPU','bandwidth')]
         non_binary_bdw_list = df[(non_binary_format,'GPU','bandwidth')]
         binary_speedup_list = []
         for ( format_bdw, non_binary_bdw ) in zip( format_bdw_list, non_binary_bdw_list ):
            try:
               binary_speedup_list.append( format_bdw / non_binary_bdw )
            except:
               binary_speedup_list.append( float('nan'))
         df[(format,'GPU','speed-up','non-binary')] = binary_speedup_list

####
# Compute speed-up of symmetric formats
def compute_symmetric_speedup( df, formats ):
   for format in formats:
      if 'Symmetric' in format:
         non_symmetric_format = format.replace( 'Symmetric ', '' )
         print( f'Adding speed-up of {format} vs {non_symmetric_format}' )
         format_times_list = df[(format,'GPU','time')]
         non_symmetric_times_list = df[(non_symmetric_format,'GPU','time')]

         symmetric_speedup_list = []
         for ( format_time, non_symmetric_time ) in zip( format_times_list, non_symmetric_times_list ):
            try:
               symmetric_speedup_list.append( non_symmetric_time / format_time  )
            except:
               symmetric_speedup_list.append(float('nan'))
         df[(format,'GPU','speed-up','non-symmetric')] = symmetric_speedup_list

def compute_speedup( df, formats ):
   compute_cusparse_speedup( df, formats )
   compute_csr_light_speedup( df, formats )
   compute_binary_speedup( df, formats )
   compute_symmetric_speedup( df, formats )

###
# Draw several profiles into one figure
def draw_profiles( formats, profiles, xlabel, ylabel, filename, legend_loc='upper right', bar='none' ):
   fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
   latexNames = []
   size = 1
   for format in formats:
      t = np.arange(profiles[format].size )
      axs.plot( t, profiles[format], '-o', ms=1, lw=1 )
      size = len( profiles[format] )
      latexNames.append( latexFormatName( format ) )
   if bar != 'none':
      #print( f'size = {size}' )
      bar_data = np.full( size, 1 )
      axs.plot( t, bar_data, '-', ms=1, lw=1.5 )
      if bar != '':
         latexNames.append( bar )

   axs.legend( latexNames, loc=legend_loc )
   axs.set_xlabel( xlabel )
   axs.set_ylabel( ylabel )
   axs.set_yscale( 'log' )
   plt.rcParams.update({
      "text.usetex": True,
      "font.family": "sans-serif",
      "font.sans-serif": ["Helvetica"]})
   plt.savefig( filename )
   plt.close(fig)


####
# Effective BW profile
def effective_bw_profile( df, formats, head_size=10 ):
   if not os.path.exists("BW-profile"):
      os.mkdir("BW-profile")
   profiles = {}
   for format in formats:
      print( f"Writing BW profile of {format}" )
      fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
      t = np.arange(df[(format,'GPU','bandwidth')].size )
      if format == 'CSR':
         df.sort_values(by=[(format,'CPU','bandwidth')],inplace=True,ascending=False)
         profiles[format] = df[(format,'CPU','bandwidth')].copy()
         axs.plot( t, df[(format,'CPU','bandwidth')], '-o', ms=1, lw=1 )
      else:
         df.sort_values(by=[(format,'GPU','bandwidth')],inplace=True,ascending=False)
         profiles[format] = df[(format,'GPU','bandwidth')].copy()
         axs.plot( t, df[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
      axs.legend( [ latexFormatName(format), 'CSR on CPU' ], loc='upper right' )
      axs.set_ylabel( 'Effective bandwidth in GB/sec' )
      plt.rcParams.update({
         "text.usetex": True,
         "font.family": "sans-serif",
         "font.sans-serif": ["Helvetica"]})
      plt.savefig( f"BW-profile/{format}.pdf")
      plt.close(fig)
      fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
      axs.set_yscale( 'log' )
      axs.plot( t, result[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
      axs.legend( [ latexFormatName(format), 'CSR on CPU' ], loc='lower left' )
      axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} performance" )
      axs.set_ylabel( 'Effective bandwidth in GB/sec' )
      plt.rcParams.update({
         "text.usetex": True,
         "font.family": "sans-serif",
         "font.sans-serif": ["Helvetica"]})
      plt.savefig( f"BW-profile/{format}-log.pdf")
      plt.close(fig)
      copy_df = df.copy()
      for f in formats:
         if not f in ['cusparse','CSR',format]:
            copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
      copy_df.to_html( f"BW-profile/{format}.html" )

   # Draw ellpack formats profiles
   current_formats = []
   xlabel = "Matrix number - sorted by particular formats effective bandwidth"
   ylabel = "Effective bandwidth in GB/sec"
   for format in formats:
      if( ( 'Ellpack' in format and not 'Binary' in format and not 'Symmetric' in format and not 'Legacy' in format ) or
          format == 'CSR' or
          format == 'cusparse' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "ellpack-profiles-bw.pdf", 'lower left', "none" )

   # Draw CSR formats profiles
   current_formats.clear()
   for format in formats:
      if( ( 'CSR' in format and not 'Binary' in format and not 'Symmetric' in format and not 'Legacy' in format and not 'Hybrid' in format ) or
          format == 'cusparse' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "csr-profiles-bw.pdf", 'lower left', 'none' )


####
# Comparison with Cusparse
def cusparse_comparison( df, formats, head_size=10 ):
   if not os.path.exists("Cusparse-bw"):
      os.mkdir("Cusparse-bw")
   ascend_df = df.copy()
   df.sort_values(by=[('cusparse','GPU','bandwidth')],inplace=True,ascending=False)
   ascend_df.sort_values(by=[('cusparse','GPU','bandwidth')],inplace=True,ascending=True)
   for format in formats:
      if not format in ['cusparse','CSR']:
         print( f"Writing comparison of {format} and cuSPARSE" )
         filtered_df = df.dropna( subset=[(format,'GPU','bandwidth','')] )
         filtered_ascend_df = ascend_df.dropna( subset=[(format,'GPU','bandwidth','')] )
         t = np.arange(filtered_df[(format,'GPU','bandwidth')].size )
         fig, axs = plt.subplots( 2, 1 )
         axs[0].plot( t, filtered_df[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].plot( t, filtered_df[('cusparse','GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].legend( [ format, 'cuSPARSE' ], loc='upper right' )
         axs[0].set_ylabel( 'Effective bandwidth in GB/sec' )
         axs[1].set_yscale( 'log' )
         axs[1].plot( t, filtered_df[(format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].plot( t, filtered_df[('cusparse','GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].legend( [ latexFormatName(format), 'cuSPARSE' ], loc='lower left' )
         axs[1].set_xlabel( 'Matrix number - sorted w.r.t. cuSPARSE performance' )
         axs[1].set_ylabel( 'Effective bandwidth in GB/sec' )
         plt.savefig( f"Cusparse-bw/{format}.pdf" )
         plt.close(fig)
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.to_html( f"Cusparse-bw/{format}.html" )

####
# Comparison with CSR on CPU
def csr_comparison( df, formats, head_size=10 ):
   if not os.path.exists("CSR-bw"):
      os.mkdir("CSR-bw")
   for device in [ 'CPU', 'GPU' ]:
      for format in formats:
         if not format in ['cusparse','CSR']:
            print( f"Writing comparison of {format} and CSR on CPU" )
            try:
               df.sort_values(by=[(format,device,'bandwidth')],inplace=True,ascending=False)
            except:
               continue
            fig, axs = plt.subplots( 2, 1 )
            t = np.arange(df[(format,device,'bandwidth')].size )
            axs[0].plot( t, df[(format,device,'bandwidth')], '-o', ms=1, lw=1 )
            axs[0].plot( t, df[('CSR','CPU','bandwidth')], '-o', ms=1, lw=1 )
            axs[0].legend( [ latexFormatName(format), 'CSR on CPU' ], loc='upper right' )
            axs[0].set_ylabel( 'Effective bandwidth in GB/sec' )
            axs[1].set_yscale( 'log' )
            axs[1].plot( t, result[(format,device,'bandwidth')], '-o', ms=1, lw=1 )
            axs[1].plot( t, result[('CSR','CPU','bandwidth')], '-o', ms=1, lw=1 )
            axs[1].legend( [ latexFormatName(format), 'CSR on CPU' ], loc='lower left' )
            axs[1].set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} performance" )
            axs[1].set_ylabel( 'Effective bandwidth in GB/sec' )
            plt.rcParams.update({
               "text.usetex": True,
               "font.family": "sans-serif",
               "font.sans-serif": ["Helvetica"]})
            plt.savefig( f"CSR-bw/{format}-{device}.pdf")
            plt.close(fig)
            copy_df = df.copy()
            for f in formats:
               if not f in ['cusparse','CSR',format]:
                  copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
            copy_df.to_html( f"CSR-bw/{format}-{device}.html" )

####
# Comparison of Legacy formats
def legacy_formats_comparison( df, formats, head_size=10 ):
   if not os.path.exists("Legacy-bw"):
      os.mkdir("Legacy-bw")
   for ref_format, legacy_format in [ ('Ellpack', 'Ellpack Legacy'),
                                    ('SlicedEllpack', 'SlicedEllpack Legacy'),
                                    ('ChunkedEllpack', 'ChunkedEllpack Legacy'),
                                    ('BiEllpack', 'BiEllpack Legacy'),
                                    ('CSR< Adaptive >', 'CSR Legacy Adaptive'),
                                    ('CSR< Scalar >', 'CSR Legacy Scalar'),
                                    ('CSR< Vector >', 'CSR Legacy Vector') ]:
      if ref_format in formats and legacy_format in formats:
         print( f"Writing comparison of {ref_format} and {legacy_format}" )
         ascend_df = df.copy()
         df.sort_values(by=[(ref_format,'GPU','bandwidth')],inplace=True,ascending=False)
         ascend_df.sort_values(by=[(ref_format,'GPU','bandwidth')],inplace=True,ascending=True)
         fig, axs = plt.subplots( 2, 1 )
         t = np.arange(df[(ref_format,'GPU','bandwidth')].size )
         axs[0].plot( t, df[(ref_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].plot( t, df[(legacy_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[0].legend( [ latexFormatName(ref_format), latexFormatName(legacy_format) ], loc='upper right' )
         axs[0].set_ylabel( 'Effective bandwidth in GB/sec' )
         axs[1].set_yscale( 'log' )
         axs[1].plot( t, df[(ref_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].plot( t, df[(legacy_format,'GPU','bandwidth')], '-o', ms=1, lw=1 )
         axs[1].legend( [ latexFormatName(ref_format), latexFormatName(legacy_format) ], loc='lower left' )
         axs[1].set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(ref_format)}  performance" )
         axs[1].set_ylabel( 'Effective bandwidth in GB/sec' )
         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
         plt.savefig( f"Legacy-bw/{ref_format}.pdf")
         plt.close(fig)
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.to_html( f"Legacy-bw/{format}.html" )

####
# Comparison of speed-up w.r.t. CSR
def csr_speedup_comparison( df, formats, head_size=10 ):
   if not os.path.exists("CSR-speed-up"):
      os.mkdir("CSR-speed-up")
   for device in ['CPU', 'GPU']:
      profiles = {}
      for format in formats:
         if not format in ['cusparse','CSR']:
            print( f"Writing comparison of speed-up of {format} compared to CSR" )
            df['tmp'] = df[(format, device,'bandwidth')]
            filtered_df=df.dropna(subset=[('tmp','','','')])
            try:
               filtered_df.sort_values(by=[(format,device,'speed-up','CSR CPU')],inplace=True,ascending=False)
            except:
               continue
            profiles[format] = filtered_df[(format,device,'speed-up','CSR CPU')].copy()
            fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
            size = len(filtered_df[(format,device,'speed-up','CSR CPU')].index)
            t = np.arange( size )
            bar = np.full( size, 1 )
            axs.plot( t, filtered_df[(format,device,'speed-up','CSR CPU')], '-o', ms=1, lw=1 )
            axs.plot( t, bar, '-', ms=1, lw=1 )
            axs.legend( [ latexFormatName(format), 'CSR CPU' ], loc='upper right' )
            axs.set_ylabel( 'Speedup' )
            axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
            plt.rcParams.update({
               "text.usetex": True,
               "font.family": "sans-serif",
               "font.sans-serif": ["Helvetica"]})
            plt.savefig( f"CSR-speed-up/{format}.pdf")
            plt.close(fig)

            fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
            axs.set_yscale( 'log' )
            axs.plot( t, filtered_df[(format,device,'speed-up','CSR CPU')], '-o', ms=1, lw=1 )
            axs.plot( t, bar, '-', ms=1, lw=1 )
            axs.legend( [ latexFormatName(format), 'CSR' ], loc='lower left' )
            axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
            axs.set_ylabel( 'Speedup' )
            plt.savefig( f"CSR-speed-up/{format}-{device}-log.pdf")
            plt.close(fig)
            copy_df = df.copy()
            for f in formats:
               if not f in ['cusparse','CSR',format]:
                  copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
            copy_df.to_html( f"CSR-speed-up/{format}-{device}.html" )


####
# Comparison of speed-up w.r.t. Cusparse
def cusparse_speedup_comparison( df, formats, head_size=10 ):
   if not os.path.exists("Cusparse-speed-up"):
      os.mkdir("Cusparse-speed-up")
   profiles = {}
   for format in formats:
      if not format in ['cusparse','CSR']:
         print( f"Writing comparison of speed-up of {format} ({latexFormatName(format)}) compared to cuSPARSE" )
         df['tmp'] = df[(format, 'GPU','bandwidth')]
         filtered_df=df.dropna(subset=[('tmp','','','')])
         filtered_df.sort_values(by=[(format,'GPU','speed-up','cusparse')],inplace=True,ascending=False)
         profiles[format] = filtered_df[(format,'GPU','speed-up','cusparse')].copy()
         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         size = len(filtered_df[(format,'GPU','speed-up','cusparse')].index)
         t = np.arange( size )
         bar = np.full( size, 1 )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','cusparse')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), 'cuSPARSE' ], loc='upper right' )
         axs.set_ylabel( 'Speedup' )
         axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
         plt.savefig( f"Cusparse-speed-up/{format}.pdf")
         plt.close(fig)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.set_yscale( 'log' )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','cusparse')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), 'cuSPARSE' ], loc='lower left' )
         axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
         axs.set_ylabel( 'Speedup' )
         plt.savefig( f"Cusparse-speed-up/{format}-log.pdf")
         plt.close(fig)
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format]:
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.to_html( f"Cusparse-speed-up/{format}.html" )

   # Draw Ellpack formats profiles
   xlabel = "Matrix number - sorted by particular formats speedup compared to cuSPARSE"
   ylabel = "Speedup"
   current_formats = []
   for format in formats:
      if( 'Ellpack' in format and not 'Symmetric' in format and not 'Binary' in format and not 'Legacy' in format ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "ellpack-profiles-cusparse-speedup.pdf", 'upper right', "cuSPARSE" )

   current_formats.clear()
   for format in formats:
      if( 'Ellpack' in format and 'Symmetric' in format and not 'Binary' in format and not 'Legacy' in format ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "symmetric-ellpack-profiles-cusparse-speedup.pdf", 'upper right', "cuSPARSE" )

   current_formats.clear()
   for format in formats:
      if( 'Ellpack' in format and not 'Symmetric' in format and 'Binary' in format and not 'Legacy' in format ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "binary-ellpack-profiles-cusparse-speedup.pdf", 'upper right', "cuSPARSE" )

   current_formats.clear()
   for format in formats:
      if( 'Ellpack' in format and 'Symmetric' in format and 'Binary' in format and not 'Legacy' in format ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "symmetric-binary-ellpack-profiles-cusparse-speedup.pdf", 'upper right', "cuSPARSE" )


   # Draw CSR formats profiles
   current_formats.clear()
   for format in formats:
      if( 'CSR' in format and not 'Symmetric' in format and not 'Binary' in format and not 'Legacy' in format and not 'Hybrid' in format and format != 'CSR' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "csr-profiles-cusparse-speedup.pdf", 'upper right', "cuSPARSE" )
   current_formats.clear()
   for format in formats:
      if( 'CSR' in format and 'Symmetric' in format and not 'Binary' in format and not 'Legacy' in format and not 'Hybrid' in format and format != 'CSR' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "symmetric-csr-profiles-cusparse-speedup.pdf", 'upper right', "cuSPARSE" )
   current_formats.clear()

   for format in formats:
      if( 'CSR' in format and not 'Symmetric' in format and 'Binary' in format and not 'Legacy' in format and not 'Hybrid' in format and format != 'CSR' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "binary-csr-profiles-cusparse-speedup.pdf", 'upper right', "cuSPARSE" )
   current_formats.clear()

   for format in formats:
      if( 'CSR' in format and 'Symmetric' in format and 'Binary' in format and not 'Legacy' in format and not 'Hybrid' in format and format != 'CSR' ):
         current_formats.append( format )
   draw_profiles( current_formats, profiles, xlabel, ylabel, "-symmetric-binary-csr-profiles-cusparse-speedup.pdf", 'upper right', "cuSPARSE" )
   current_formats.clear()

####
# Comparison of binary matrices
def binary_matrices_comparison( df, formats, head_size = 10 ):
   if not os.path.exists("Binary-speed-up"):
      os.mkdir("Binary-speed-up")
   for format in formats:
      if 'Binary' in format:
         non_binary_format = format.replace('Binary ','')
         print( f"Writing comparison of speed-up of {format} vs {non_binary_format}" )
         #df['tmp'] = df[(format, 'GPU','speed-up','non-binary')]
         filtered_df=df.dropna(subset=[(format, 'GPU','speed-up','non-binary')]) #('tmp','','','')])
         #print( f"{format} -> {filtered_df[(format,'GPU','speed-up','non-binary')]}" )
         ascend_df = filtered_df.copy()
         filtered_df.sort_values(by=[(format,'GPU','speed-up','non-binary')],inplace=True,ascending=False)
         ascend_df.sort_values(by=[(format,'GPU','speed-up','non-binary')],inplace=True,ascending=True)
         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         size = len(filtered_df[(format,'GPU','speed-up','non-binary')].index)
         t = np.arange( size )
         bar = np.full( size, 1 )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','non-binary')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), latexFormatName(non_binary_format) ], loc='upper right' )
         axs.set_ylabel( 'Speedup' )
         axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
         # for Palatino and other serif fonts use:
         #plt.rcParams.update({
         #   "text.usetex": True,
         #   "font.family": "serif",
         #   "font.serif": ["Palatino"],
         #})
         plt.savefig( f"Binary-speed-up/{format}.pdf")
         plt.close(fig)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.set_yscale( 'log' )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','non-binary')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), latexFormatName(non_binary_format) ], loc='upper right' )
         axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
         axs.set_ylabel( 'Speedup' )
         plt.savefig( f"Binary-speed-up/{format}-log.pdf")
         plt.close(fig)
         #head_df = filtered_df.head( head_size )
         #bottom_df = ascend_df.head( head_size )
         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format,non_binary_format]:
               #print( f"Droping {f}..." )
               #head_df.drop( labels=f, axis='columns', level=0, inplace=True )
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         #head_df.to_html( f"Binary-speed-up/{format}-head.html" )
         copy_df.to_html( f"Binary-speed-up/{format}.html" )

####
# Comparison of symmetric matrices
def symmetric_matrices_comparison( df, formats, head_size = 10 ):
   if not os.path.exists("Symmetric-speed-up"):
      os.mkdir("Symmetric-speed-up")
   for format in formats:
      if 'Symmetric' in format:
         non_symmetric_format = format.replace('Symmetric ','')
         print( f"Writing comparison of speed-up of {format} vs {non_symmetric_format}" )
         #df['tmp'] = df[(format, 'GPU','speed-up','non-symmetric')]
         filtered_df=df.dropna(subset=[(format, 'GPU','speed-up','non-symmetric')]) #('tmp','','','')])
         #ascend_df = filtered_df.copy()
         #print( f"{format} -> {filtered_df[(format,'GPU','speed-up','non-symmetric')]}" )
         filtered_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,ascending=False)
         #ascend_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,ascending=True)

         cusparse_filtered_df=df.dropna(subset=[(format, 'GPU','speed-up','cusparse')]) #('tmp','','','')])
         cusparse_filtered_df.sort_values(by=[(format,'GPU','speed-up','cusparse')],inplace=True,ascending=False)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         size = len(filtered_df[(format,'GPU','speed-up','non-symmetric')].index)
         t = np.arange( size )
         bar = np.full( size, 1 )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','non-symmetric')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), latexFormatName(non_symmetric_format) ], loc='upper right' )
         axs.set_ylabel( 'Speedup' )
         axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
         plt.savefig( f"Symmetric-speed-up/{format}.pdf")
         plt.close(fig)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.set_yscale( 'log' )
         axs.plot( t, filtered_df[(format,'GPU','speed-up','non-symmetric')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), latexFormatName(non_symmetric_format) ], loc='lower left' )
         axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
         axs.set_ylabel( 'Speedup' )
         plt.savefig( f"Symmetric-speed-up/{format}-log.pdf")
         plt.close(fig)
         #head_df = filtered_df.head( head_size )
         #bottom_df = ascend_df.head( head_size )

         size = len(cusparse_filtered_df[(format,'GPU','speed-up','cusparse')].index)
         t = np.arange( size )
         bar = np.full( size, 1 )
         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.plot( t, cusparse_filtered_df[(format,'GPU','speed-up','cusparse')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), 'cuSPARSE' ], loc='upper right' )
         axs.set_ylabel( 'Speedup' )
         axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
         plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
         plt.savefig( f"Symmetric-speed-up/{format}-cusparse.pdf")
         plt.close(fig)

         fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
         axs.set_yscale( 'log' )
         axs.plot( t, cusparse_filtered_df[(format,'GPU','speed-up','cusparse')], '-o', ms=1, lw=1 )
         axs.plot( t, bar, '-', ms=1, lw=1 )
         axs.legend( [ latexFormatName(format), 'cuSPARSE' ], loc='lower left' )
         axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
         axs.set_ylabel( 'Speedup' )
         plt.savefig( f"Symmetric-speed-up/{format}-cusparse-log.pdf")
         plt.close(fig)


         copy_df = df.copy()
         for f in formats:
            if not f in ['cusparse','CSR',format,non_symmetric_format]:
               #print( f"Droping {f}..." )
               #head_df.drop( labels=f, axis='columns', level=0, inplace=True )
               copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
         #head_df.to_html( f"Symmetric-speed-up/{format}-head.html" )
         copy_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,ascending=False)
         copy_df.to_html( f"Symmetric-speed-up/{format}.html" )
         #copy_df.sort_values(by=[(format,'GPU','speed-up','non-symmetric')],inplace=True,descending=True)
         #copy_df.to_html( f"Symmetric-speed-up/{format}-sort.html" )

####
# Comparison of speed-up w.r.t. LightSpMV
def csr_light_speedup_comparison( df, head_size=10 ):
   format = 'CSR< Light > Automatic Light'
   print( f"Writing comparison of speed-up of CSR Light compared to LightSPMV" )
   df['tmp'] = df[(format, 'GPU','bandwidth')]
   filtered_df=df.dropna(subset=[('tmp','','','')])
   ascend_df = filtered_df.copy()
   filtered_df.sort_values(by=[(format,'GPU','speed-up','LightSpMV Vector')],inplace=True,ascending=False)
   ascend_df.sort_values(by=[(format,'GPU','speed-up','LightSpMV Vector')],inplace=True,ascending=True)
   fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
   size = len(filtered_df[(format,'GPU','speed-up','LightSpMV Vector')].index)
   t = np.arange( size )
   bar = np.full( size, 1 )
   axs.plot( t, filtered_df[(format,'GPU','speed-up','LightSpMV Vector')], '-o', ms=1, lw=1 )
   axs.plot( t, bar, '-', ms=1, lw=1 )
   axs.legend( [ latexFormatName(format), 'LightSpMV' ], loc='upper right' )
   axs.set_ylabel( 'Speedup' )
   axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
   plt.rcParams.update({
      "text.usetex": True,
      "font.family": "sans-serif",
      "font.sans-serif": ["Helvetica"]})
   # for Palatino and other serif fonts use:
   #plt.rcParams.update({
   #   "text.usetex": True,
   #   "font.family": "serif",
   #   "font.serif": ["Palatino"],
   #})
   plt.savefig( f"LightSpMV-speed-up.pdf")
   plt.close(fig)

   fig, axs = plt.subplots( 1, 1, figsize=(6,4) )
   axs.set_yscale( 'log' )
   axs.plot( t, filtered_df[(format,'GPU','speed-up','LightSpMV Vector')], '-o', ms=1, lw=1 )
   axs.plot( t, bar, '-', ms=1, lw=1 )
   axs.legend( [ latexFormatName(format), 'LightSpMV' ], loc='lower left' )
   axs.set_xlabel( f"Matrix number - sorted w.r.t. {latexFormatName(format)} speed-up" )
   axs.set_ylabel( 'Speedup' )
   plt.savefig( f"LightSpMV-speed-up-log.pdf")
   plt.close(fig)
   #head_df = filtered_df.head( head_size )
   #bottom_df = ascend_df.head( head_size )
   copy_df = df.copy()
   for f in formats:
      if not f in ['cusparse','CSR',format]:
         #print( f"Droping {f}..." )
         #head_df.drop( labels=f, axis='columns', level=0, inplace=True )
         copy_df.drop( labels=f, axis='columns', level=0, inplace=True )
   #head_df.to_html( f"LightSpMV-speed-up-head.html" )
   copy_df.to_html( f"LightSpMV-speed-up-bottom.html" )

def write_colormap( file, max_bw, size, x_position, y_position, standalone = False ):
   if standalone:
      file.write( '\\documentclass{standalone}\n' )
      file.write( '\\usepackage[utf8]{inputenc}\n' )
      file.write( '\\usepackage{tikz}\n' )
      file.write( '\\begin{document}\n' )
      file.write( '\\begin{tikzpicture}\n' )
   i = 0
   x = x_position
   while i <= max_bw:
      y = y_position + i / max_bw * size
      r, g, b = color_map(i, max_bw, map=heatmap)
      file.write( f'\\definecolor{{color_hm_{i}}}{{rgb}}{{ {r}, {g}, {b} }}; \n' )
      file.write( f'\\filldraw[color_hm_{i}] ({x},{y}) circle (2pt); \n' )
      i = i + 5
   i = 0
   while i <= max_bw:
      y = y_position + i / max_bw * size
      file.write( f'\\filldraw[black] ({x},{y}) circle (1pt) node[anchor=west] {{{i}}}; \n' )
      i = i + 400

   if standalone:
      file.write( '\\end{tikzpicture}\n' )
      file.write( '\\end{document}\n' )

def write_performance_circle_latex_base( file_name ):
   file = open( f'{file_name}-base.tex', 'w')
   file.write( '\\documentclass{standalone}\n' )
   file.write( '\\usepackage[utf8]{inputenc}\n' )
   file.write( '\\usepackage{tikz}\n' )
   file.write( '\\begin{document}\n' )
   file.write( '\\begin{tikzpicture}\n' )
   file.write( f'\\input{{{file_name}.tex}}\n' )
   file.write( '\\end{tikzpicture}\n' )
   file.write( '\\end{document}\n' )

#####
# Draw performance circle in tikz
def write_performance_circle( df, formats, circle_formats, file_name, scale=1, with_color_map = False ):
   write_performance_circle_latex_base( file_name )
   file = open( f'{file_name}.tex', 'w')
   formats_number = 0
   for format in circle_formats:
      if format in formats:
         formats_number += 1

   format_idx = 0
   pos_x = 5 * scale
   pos_y = 5 * scale
   rad = 5 * scale
   formats_pos_x = {}
   formats_pos_y = {}
   for format in circle_formats:
      if format in formats:
         format_angle = math.pi/2 - 2*math.pi/formats_number*format_idx - math.pi / formats_number
         if format_angle < 0:
            format_angle = 2*math.pi + format_angle
         x = pos_x + rad*math.cos( format_angle )
         y = pos_y + rad*math.sin( format_angle )
         formats_pos_x[ format ] = x
         formats_pos_y[ format ] = y
         anchor = ''
         if format_angle <= math.pi * 1/4  or format_angle > math.pi * 7/4:
            anchor = 'west'
         if format_angle <= math.pi * 3/4 and format_angle > math.pi * 1/4:
            anchor = 'south'
         if format_angle <= math.pi * 5/4 and format_angle > math.pi * 3/4:
            anchor = 'east'
         if format_angle <= math.pi * 7/4 and format_angle > math.pi * 5/4:
            anchor = 'north'
         #print( f'{format_angle} : {format} -> {anchor} \n' )
         file.write( f'\\filldraw[black] ({x},{y}) circle (2pt) node[anchor={anchor}]{{{latexFormatName(format)}}}; \n' )
         div_angle = format_angle + math.pi / formats_number
         div_x = pos_x + rad*math.cos( div_angle )
         div_y = pos_y + rad*math.sin( div_angle )
         file.write( f'\\draw [dashed] ({div_x},{div_y}) -- ({pos_x},{pos_y}); \n')
         format_idx += 1
   formats_count = format_idx
   line_idx=0
   elim = 0
   while line_idx < len(df.index):
      #matrixName = df.iloc[line_idx]['Matrix name']
      sum_bw = 0
      formats_bw = {}
      max_bw = 0
      for format in circle_formats:
         if format in formats:
            format_bw = df.iloc[line_idx][(format,'GPU','bandwidth','')]
            formats_bw[ format ] = format_bw
            #print( f'{matrixName} {format} -> {format_bw}')
            #if format_bw > max_bw:
            sum_bw = sum_bw + format_bw
            if format_bw > max_bw:
               max_bw = format_bw
      for format in circle_formats:
         if format in formats:
            formats_bw[ format ] = formats_bw[ format ] / sum_bw
      format_pos_x = 0
      format_pos_y = 0
      for format in circle_formats:
         if format in formats:
            format_pos_x = format_pos_x + formats_pos_x[ format ] * formats_bw[ format ]
            format_pos_y = format_pos_y + formats_pos_y[ format ] * formats_bw[ format ]
      if( format_pos_x == format_pos_x  and format_pos_y == format_pos_y ):  # check for NaN
         r, g, b = color_map(max_bw, 1200, map=heatmap)
         file.write( f'\\definecolor{{color_{line_idx}}}{{rgb}}{{ {r}, {g}, {b} }} \n' )
         file.write( f'\\filldraw[color_{line_idx},opacity=0.75] ({format_pos_x},{format_pos_y}) circle (1pt); \n' )
      else:
         elim = elim + 1
      line_idx += 1
   if with_color_map:
      write_colormap( file, 1200, 5, 13*scale, 1.5*scale, standalone=False )
   os.system( f'pdflatex {file_name}-base.tex' )
   print( f'Eliminated formats: {elim}')

####
# Parse input file
print( "Parsing input file...." )
with open('sparse-matrix-benchmark.log') as f:
    d = json.load(f)
input_df = json_normalize( d, record_path=['results'] )
#input_df.to_html( "orig-pandas.html" )

formats = list(set( input_df['format'].values.tolist() )) # list of all formats in the benchmark results
formats.remove('CSR< Light > Automatic')
formats.remove('Binary CSR< Light > Automatic')
formats.remove('Symmetric CSR< Light > Automatic')
formats.remove('Symmetric Binary CSR< Light > Automatic')
formats.append('TNL Best')
multicolumns, df_data = get_multiindex( input_df, formats )

print( "Converting data..." )
result = convert_data_frame( input_df, multicolumns, df_data, 0, 20000 )
compute_speedup( result, formats )

result.replace( to_replace=' ',value=np.nan,inplace=True)

####
# Make data analysis
def processDf( df, formats, head_size = 10 ):
   print( "Writting to HTML file..." )
   df.to_html( f'output.html' )

   # Generate tables and figures
   effective_bw_profile( df, formats, head_size )
   cusparse_comparison( df, formats, head_size )
   csr_comparison( df, formats, head_size )
   legacy_formats_comparison( df, formats, head_size )
   csr_speedup_comparison( df, formats, head_size )
   cusparse_speedup_comparison( df, formats, head_size )
   binary_matrices_comparison( df, formats, head_size )
   symmetric_matrices_comparison( df, formats, head_size )
   csr_light_speedup_comparison( df, head_size )

   best = df[('TNL Best','GPU','format')].tolist()
   best_formats = list(set(best))
   sum = 0
   for format in formats:
      if( not 'Binary' in format and
          not 'Symmetric' in format and
          not 'Legacy' in format and
          not 'LightSpMV' in format and
          not 'TNL Best' in format ):
         cases = best.count(format)
         print( f'{format} is best in {cases} cases.')
         sum += cases
   print( f'Total is {sum}.' )
   print( f'Best formats {best_formats}.')
   write_performance_circle( df, formats,
         ['cusparse', 'Ellpack', 'SlicedEllpack', 'ChunkedEllpack', 'BiEllpack', 'CSR< Scalar >', 'CSR< Adaptive >', 'CSR< Vector >', 'CSR< Light > Automatic Light'],
         'performance-graph' )

   scale = 0.6
   aux_df = df
   aux_df.sort_values(by=[('SlicedEllpack','GPU','bandwidth')],inplace=True,ascending=True)
   write_performance_circle( aux_df, formats, ['Ellpack', 'ChunkedEllpack', 'SlicedEllpack' ], 'performance-graph-ellpacks-1', scale, with_color_map = False )
   write_performance_circle( aux_df, formats, ['BiEllpack', 'ChunkedEllpack', 'SlicedEllpack',  ], 'performance-graph-ellpacks-2', scale, with_color_map = True )
   #write_performance_circle( df, formats, ['CSR< Scalar >', 'CSR< Adaptive >', 'CSR< Vector >', 'CSR< Light > Automatic Light'], 'performance-graph-csr-1' )
   aux_df.sort_values(by=[('CSR< Light > Automatic Light','GPU','bandwidth')],inplace=True,ascending=True)
   write_performance_circle( aux_df, formats, ['CSR< Scalar >', 'CSR< Vector >', 'CSR< Light > Automatic Light'], 'performance-graph-csr-1', scale, with_color_map = False )
   write_performance_circle( aux_df, formats, ['CSR< Adaptive >', 'CSR< Vector >', 'CSR< Light > Automatic Light'], 'performance-graph-csr-2', scale, with_color_map = False )
   aux_df.sort_values(by=[('cusparse','GPU','bandwidth')],inplace=True,ascending=True)
   write_performance_circle( aux_df, formats, ['cusparse', 'SlicedEllpack', 'ChunkedEllpack' ], 'performance-graph-cusparse-ellpacks', scale, with_color_map = False )
   write_performance_circle( aux_df, formats, ['cusparse', 'CSR< Vector >', 'CSR< Light > Automatic Light'], 'performance-graph-cusparse-csr-1', scale, with_color_map = False )
   write_performance_circle( aux_df, formats, ['cusparse', 'CSR< Adaptive >', 'CSR< Light > Automatic Light'], 'performance-graph-cusparse-csr-2', scale, with_color_map = True )
   write_performance_circle( aux_df, formats, ['cusparse', 'CSR< Scalar >', 'CSR< Light > Automatic Light'], 'performance-graph-cusparse-csr-3', scale, with_color_map = False )
   write_performance_circle( aux_df, formats, ['cusparse', 'SlicedEllpack', 'CSR< Light > Automatic Light'], 'performance-graph-cusparse-csr-ellpack', scale, with_color_map = True )

head_size = 25
if not os.path.exists( 'general' ):
   os.mkdir( 'general' )
os.chdir( 'general' )
processDf( result, formats, head_size )
os.chdir( '..' )

#for rows_count in [ 10, 100, 1000, 10000, 100000, 1000000, 10000000 ]:
#   filtered_df = result[ result['rows'].astype('int32') <= rows_count ]
#   if not os.path.exists(f'rows-le-{rows_count}'):
#      os.mkdir( f'rows-le-{rows_count}')
#   os.chdir( f'rows-le-{rows_count}')
#   processDf( filtered_df, formats, head_size )
#   os.chdir( '..' )

#for rows_count in [ 10, 100, 1000, 10000, 100000, 1000000, 10000000 ]:
#   filtered_df = result[ result['rows'].astype('int32') >= rows_count ]
#   if not os.path.exists(f'rows-ge-{rows_count}'):
#      os.mkdir( f'rows-ge-{rows_count}')
#   os.chdir( f'rows-ge-{rows_count}')
#   processDf( filtered_df, formats, head_size )
#   os.chdir( '..' )
