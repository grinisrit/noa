#!/usr/bin/env python3

import warnings
warnings.warn("The CustomLogging format for TNL benchmarks is deprecated. Please switch your benchmark "
              "to JsonLogging and use the tnl-benchmark-to-html.py script for post-processing.",
              DeprecationWarning)

import collections

try:
    import pandas
    pandas.set_option('display.max_columns', 100)
    pandas.set_option('display.max_rows', 1000)
    pandas.set_option('display.width', 150)
except ImportError:
    raise ImportError("Please make sure that the python3-pandas package is installed.")


def try_numeric(value):
    try:
       return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

class TableColumn:
    def __init__(self, level, data, parentPath=None):
        self.subcolumns = []
        self.height = 0
        self.numberOfSubcolumns = 0
        self.rowspan = 0

        self.level = level
        # splitting with spaces around ':' is necessary, because the label can contain ':' too
        # (e.g. C++ types like Matrices::CSR)
        dataSplit = data.split( ' : ', 1 )
        self.label = dataSplit[ 0 ].strip()
        if len(dataSplit) == 2:
            self.attributes = dataSplit[1]
        else:
            self.attributes = ""

        if parentPath is None:
            self.path = []
        else:
            # make a copy!
            self.path = parentPath[:]
        self.path.append(self.label)

    def insertSubcolumn( self, level, label ):
        if level == self.level + 1:
            self.subcolumns.append( TableColumn( level, label, self.path ) )
        if level > self.level + 1:
            self.subcolumns[ -1 ].insertSubcolumn( level, label )

    def pickLeafColumns( self, leafColumns ):
        if len( self.subcolumns ) == 0:
            leafColumns.append( self )
        else:
            for subcolumn in self.subcolumns:
                subcolumn.pickLeafColumns( leafColumns )

    def __repr__(self):
        return "<TableColumn(label={}, subcolumns={})>".format(self.label, [col.label for col in self.subcolumns])


class LogParser:
    def readFile(self, logFileName):
        logFile = open(logFileName, 'r')

        # read file by lines
        lines = logFile.readlines()

        # drop comments and blank lines
        lines = [line for line in lines if line.strip() and not line.startswith("#")]

        # drop anything before the first metadata block
        while len(lines) > 0 and not lines[0].startswith(":"):
            lines.pop(0)

        while len(lines) > 0:
            metadata = []
            while len(lines) > 0 and lines[0].startswith(":"):
                metadata.append(lines.pop(0))
            metadata = self.parseMetadata(metadata)

            table = []
            while len(lines) > 0 and not lines[0].startswith(":"):
                table.append(lines.pop(0))
            tableColumns, tableRows = self.parseTable(table)

            df = self.getDataframe(tableColumns, tableRows)
            df = df.sort_index()

            yield metadata, df

    @staticmethod
    def parseMetadata(lines):
        metadata = {}
        for line in lines:
            line = line[1:]
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
        return metadata

    def parseTable(self, lines):
        self.tableColumns = collections.OrderedDict()
        rows = []
        while len(lines) > 0:
            header = []
            while len(lines) > 0 and lines[0].startswith("!"):
                header.append(lines.pop(0))
            body = []
            while len(lines) > 0 and not lines[0].startswith("!"):
                body.append(lines.pop(0))
            rows.append(self.parseTableRow(header, body))
        return self.tableColumns, rows

    def parseTableRow(self, header, body):
        columns = []
        for line in header:
            data = line.lstrip("!")
            level = len(line) - len(data)
            label = data.strip()
            if level == 1:
                columns.append( TableColumn( 1, label ) )
            if level > 1:
                columns[ -1 ].insertSubcolumn( level, label )

        # merge columns of this block with the previously parsed columns
        self.mergeColumns(columns)

        # pick leaf columns (data will be added here)
        leafColumns = self.pickLeafColumns(columns)

        # elements of the table row corresponding to the header just parsed
        elements = [line.strip() for line in body]

        if len(elements) != len(leafColumns):
            raise Exception("Error in the table format: header has {} leaf columns, but the corresponding row has {} elements.".format(len(leafColumns), len(elements)))

        row = collections.OrderedDict()
        for element, column in zip(elements, leafColumns):
            path = tuple(column.path)
            row[path] = element
        return row

    def pickLeafColumns(self, columns):
        leafColumns = []
        for column in columns:
            column.pickLeafColumns(leafColumns)
        return leafColumns

    def mergeColumns(self, columns):
        for col in columns:
            path = tuple(col.path)
            if path in self.tableColumns:
                # merge all column attributes
                self.tableColumns[path].attributes += " " + col.attributes
                # merge new subcolumns
                currentSubPaths = [tuple(col.path) for col in self.tableColumns[path].subcolumns]
                for subcol in col.subcolumns:
                    if tuple(subcol.path) not in currentSubPaths:
                        self.tableColumns[path].subcolumns.append(subcol)
            else:
                self.tableColumns[path] = col
            self.mergeColumns(col.subcolumns)

    @staticmethod
    def getDataframe(tableColumns, tableRows):
        # names of the index and data columns
        index_names = [k[0] for k, v in tableColumns.items() if not v.subcolumns and len(k) == 1]
        column_names = [k for k, v in tableColumns.items() if not v.subcolumns and len(k) > 1]

        values = collections.OrderedDict()
        for row in tableRows:
            # split row into index and data columns
            idx_itms = {}
            col_val = {}
            for k, v in row.items():
                if len(k) == 1 and k[0] in index_names:
                    idx_itms[k[0]] = try_numeric(v)
                else:
                    col_val[k] = try_numeric(v)

            # construct the index tuple
            idx = []
            for i in index_names:
                idx.append(idx_itms[i])
            idx = tuple(idx)

            # record the values
            values.setdefault(idx, {})
            values[idx].update(col_val)

        # create empty dataframe
        columns = pandas.MultiIndex.from_tuples(column_names)
        index = pandas.MultiIndex.from_tuples(values.keys(), names=index_names)
        df = pandas.DataFrame(index=index, columns=columns)

        # add data to the dataframe
        for idx, d in values.items():
            for col, val in d.items():
                df.loc[idx, col] = val

        return df
