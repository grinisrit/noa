#!/usr/bin/python3

__all__ = [
    "dict_to_html_table",
    "get_benchmark_metadata",
    "get_benchmark_dataframe",
    "gen_dataframes_per_operation",
]

import os.path
import json
import pandas

def dict_to_html_table(data):
    html = "<table border=1>\n"
    html += "<tbody>\n"
    for key in sorted(data.keys()):
        html += f"\t<tr><td>{key}</td><td>{data[key]}</td></tr>\n"
    html += "</tbody>\n"
    html += "</table>\n"
    return html

def get_benchmark_metadata(filename):
    """
    Reads metadata of the benchmark in the given file.

    :param str filename: path of the file with metadata or benchmark results.
        - If it ends with ".metadata.json", metadata is read from that file.
        - Otherwise, the extension is first replaced with ".metadata.json".
    :returns: dict as returned by json.load, or None if the file does not exist.
    """
    if not filename.endswith(".metadata.json"):
        filename = os.path.splitext(filename)[0] + ".metadata.json"
    if os.path.isfile(filename):
        print(f"Parsing metadata from file {filename}")
        return json.load(open(filename, "r"))
    print(f"Metadata file {filename} does not exist")
    return None

def get_benchmark_dataframe(logFile):
    """
    Get pandas dataframe with benchmark results stored in the given log file.

    :param logFile: path to the log file
    :returns: pandas.DataFrame instance
    """
    print(f"Parsing input file {logFile}")
    df = pandas.read_json(open(logFile, "r"), orient="records", lines=True)

    # convert "N/A" in the speedup column to nan
    if "speedup" in df.columns:
        df["speedup"] = pandas.to_numeric(df["speedup"], errors="coerce")

    return df

def gen_dataframes_per_operation(logFile, header_elements=None):
    """
    Reads benchmark results stored in the given log file and splits them into
    multiple dataframes according to the "operation" column.

    Various post-processing steps are done on each partial dataframe:
    - columns with only NaN values are removed
    - the operation column is removed
    - the "index" and "columns" of the dataframe are set:
        - if header_elements are given, they are set as "columns" and everything
          else is used for the index
        - otherwise, all columns in the dataframe before "time" are used for
          the index, and the remaining columns (starting with "time") stay as
          "columns"
    - the "performer" column is set as the last column of the index
    - note that the index is not explicitly sorted, so data is ordered as in the
      input file

    :param logFile: path to the log file
    :yields: pairs of (str, pandas.DataFrame) object, where the str denotes the
             particular operation name
    """
    main_df = get_benchmark_dataframe(logFile)

    # check if there is at least one operation
    if "operation" not in main_df.columns:
        yield "Dummy operation", main_df
        return

    # extract all benchmark operations, preserve their order as found in the dataframe
    operations = []
    for op in main_df["operation"]:
        if op not in operations:
            operations.append(op)

    # set operation as index
    main_df = main_df.set_index("operation")

    # if header_elements was not provided, we assume that "time" and all following columns
    # are benchmark results, and all preceding columns are metadata columns that will be
    # set as index of the dataframe
    if header_elements is None:
        header_elements = list(main_df.columns)
        header_elements = header_elements[header_elements.index("time"):]
        # FIXME: the "rows" and "columns" (in the gemv operation) are parsed after the correct header elements, because the preceding operations don't have these metadata columns
        # TODO: each benchmark should record the header elements in the metadata file
        header_elements = [e for e in header_elements if e not in ["rows", "columns"]]

    # emit one df per operation
    for op in operations:
        df = main_df.loc[op]
        # remove columns with only NaNs
        df = df.dropna(axis=1, how="all")
        # remove the operation column (index)
        df = df.reset_index(drop=True)
        # prepare index_columns and make sure that performer is the last
        index_columns = [c for c in df.columns if c not in header_elements and c != "performer"]
        index_columns.append("performer")
        # set new index for the df: all columns except header_elements
        df = df.set_index(index_columns)
        # emit a pair (op, df)
        yield op, df
