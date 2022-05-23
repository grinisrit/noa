#!/usr/bin/python3

import sys
import os.path
import matplotlib.pyplot as plt

from TNL.BenchmarkLogs import *
from TNL.BenchmarkPlots import *

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print(f"""\
usage: {sys.argv[0]} FILE.log [OUTPUT.html]

where FILE.log contains one JSON record per line,
and OUTPUT.html is the output file name (by default, OUTPUT=FILE).
""", file=sys.stderr)
    sys.exit(1)

logFile = sys.argv[1]
if len(sys.argv) > 2:
    htmlFile = sys.argv[2]
else:
    htmlFile = os.path.splitext(logFile)[0] + ".html"


metadata = get_benchmark_metadata(logFile)
if metadata is not None and "title" in metadata:
    title = metadata["title"]
else:
    title = os.path.splitext(os.path.basename(logFile))[0]
dataframes = list(gen_dataframes_per_operation(logFile))

print(f"Writing output to {htmlFile}")
with open(htmlFile, 'w') as f:
    print("<html>", file=f)
    # add some basic style
    print("""\
<head>
<meta charset="UTF-8">
<style>
    h1, h2 { border-bottom: solid 1px lightgray; }
    table { border-collapse: collapse; }
    table.benchmark td { text-align: end; }
    th, td { padding: 2px; }
</style>
</head>
<body>""", file=f)

    print(f"<h1>{title}</h1>", file=f)
    if metadata is not None:
        print(dict_to_html_table(metadata), file=f)

    # create a TOC
    print(f"<h2>Table of contents</h2>", file=f)
    print("<ol>", file=f)
    for op, df in dataframes:
        id = op.replace(" ", "_")
        print(f"<li><a href=\"#{id}\">{op}</a></li>", file=f)
    print("</ol>", file=f)

    # formatters for specific columns of the table
    formatters = {
        "stddev": lambda value: f"{value:e}",
        "bandwidth": lambda value: f"{value:.3f}",
        "speedup": lambda value: f"{value:.3f}",
    }

    for op, df in dataframes:
        # section heading
        id = op.replace(" ", "_")
        print(f"<h2 id=\"{id}\">{op}</h2>", file=f)
        # table
        print(df.to_html(classes="benchmark", formatters=formatters), file=f)

        # graphs
        size_name = None
        if "size" in df.index.names:
            size_name = "size"
        elif "DOFs" in df.index.names:
            size_name = "DOFs"
        if size_name is not None:
            fig, ax = plot_bandwidth_vs_size(df, size_name)
            print(get_image_html_tag(fig, format="png"), file=f)
            plt.close(fig)

        # heatmaps
        if "rows" in df.index.names and "columns" in df.index.names:
            for fig, ax in heatmaps_bandwidth(df):
                print(get_image_html_tag(fig, format="png"), file=f)
                plt.close(fig)

    print("</body>", file=f)
    print("</html>", file=f)
