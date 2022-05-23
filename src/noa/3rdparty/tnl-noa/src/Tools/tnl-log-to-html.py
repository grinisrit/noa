#!/usr/bin/env python3

import warnings
warnings.warn("The CustomLogging format for TNL benchmarks is deprecated. Please switch your benchmark "
              "to JsonLogging and use the tnl-benchmark-to-html.py script for post-processing.",
              DeprecationWarning)

import sys

from TNL.LogParser import LogParser

def metadata_to_html(metadata):
    html = "<h2>{}</h2>\n".format(metadata.get("title"))
    html += "<table border=1>\n"
    html += "<tbody>\n"
    for key in sorted(metadata.keys()):
        html += "    <tr><td>{}</td><td>{}</td></tr>\n".format(key, metadata[key])
    html += "</tbody>\n"
    html += "</table>\n"
    return html

def convertLogToHtml(logFileName, htmlFileName):
    # init HTML text
    html = "<html>\n"
    html += "<body>\n"

    parser = LogParser()

    print("Processing file", logFileName)
    for metadata, df in parser.readFile(logFileName):
        html += metadata_to_html(metadata)
        html += df.to_html()

    html += "</body>\n"
    html += "</html>\n"

    print("Writing output to", htmlFileName)
    htmlFile = open(htmlFileName, 'w')
    htmlFile.write(html)
    htmlFile.close()


arguments = sys.argv[ 1: ]
logFile = arguments[ 0 ]
if len(arguments) > 1:
    htmlFile = arguments[ 1 ]
else:
    htmlFile = logFile.rsplit(".", maxsplit=1)[0] + ".html"

convertLogToHtml(logFile, htmlFile)
