from enum import Enum
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import plac
import os

import logging as log
log.basicConfig(
    format='%(asctime)s %(levelname)s [%(module)s:%(lineno)d] -- %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S.%03d',
    level=log.INFO)


instrument_cols = ['instrument_id', 'instrument_type', 'strike', 'maturity']


def resample_data(input_dir, output_dir, freq):
    data_files = [f for f in sorted(os.listdir(input_dir)) if os.path.isfile(os.path.join(input_dir, f))]
    log.info(f'Found {len(data_files)} files in {input_dir}')
    log.info(f'Sampling with {freq} frequency')
    for data_file in data_files:
        books = pd.read_hdf(os.path.join(input_dir, data_file))
        books = books[(books.bid_amount_total > 0.) & (books.ask_amount_total > 0.)]
        books['dt'] = pd.to_datetime(books.timestamp, unit='ms')
        books = books.drop(columns='timestamp')
        df = []
        for _,v in books.groupby(instrument_cols):
            v = v.reset_index()
            v = v.drop(columns='index')
            v = v.drop_duplicates('dt')
            v = v.set_index('dt').sort_index()
            v = v.resample(freq).ffill().dropna()
            df.append(v)
        df = pd.concat(df).sort_index()
        df['date'] = df.index.map(lambda s : s.date())
        for date, v in df.groupby('date'):
            v = v.drop(columns='date')
            dt_fmt = f'{date.strftime('%Y%m%d')}'
            store_path = os.path.join(output_dir, f'{dt_fmt}_{freq}.hdf')
            if os.path.exists(store_path):
                log.info(f'Found {store_path}, updating...')
                store = pd.read_hdf(store_path)
                pd.concat([store, v])\
                    .reset_index()\
                    .drop_duplicates(['dt']+ instrument_cols)\
                    .set_index('dt').sort_index()\
                    .to_hdf(store_path, key='quotes')                                                              
            else:
                log.info(f'Creating {store_path}...')
                v.to_hdf(store_path, key='quotes')

    
@plac.annotations(
    input_dir=('path to input data directory', 'option', 'i', str),
    output_dir=('path to data directory', 'option', 'o', str),
    freq=('sampling frequency', 'option', 'fr', str)
)
def main(input_dir, output_dir, freq):
    log.info(f'Writing to {output_dir}.')
    os.makedirs(output_dir, exist_ok=True)
    resample_data(input_dir, output_dir, freq)
    
    
if __name__ == '__main__':
    plac.call(main)
