from pandas import DataFrame
import numpy as np

INSERT_MULTIPLE_DATA_HEADER_TEMPLATE: str = """INSERT INTO {} ("""
INSERT_MULTIPLE_DATA_COLUMNS_TEMPLATE: str = """{},"""
INSERT_MULTIPLE_DATA_AFTER_COLUMNS_TEMPLATE: str = """) VALUES """

# TODO: check correct work
def INSERT_MULTIPLE_DATA_VALUES_SYMBOL_TEMPLATE(dataframe: DataFrame) -> str:
    query = ""
    query += ''.join(map(lambda x: INSERT_MULTIPLE_DATA_COLUMNS_TEMPLATE.format(x), dataframe.columns))[:-1]
    query += INSERT_MULTIPLE_DATA_AFTER_COLUMNS_TEMPLATE
    for index in range(dataframe.shape[0]):
        values = dataframe.iloc[index].values
        query += "("
        query += ''.join(map(lambda x: '{},'.format(x) if type(x) != np.str_ else '"{}",'.format(x), values))[:-1] + '),'
    return query[:-1]

