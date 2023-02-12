import pandas as pd


df: pd.DataFrame = pd.read_hdf("OrderBookSubscriptionCONSTANT_TABLE_DEPTH_10.h5")
print(df.columns)
print(df)
print(df.shape)
print(df["TIMESTAMP_VALUE"].values[0])
