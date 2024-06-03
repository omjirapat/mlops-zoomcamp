import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: pd.DataFrame

    # URL of the Parquet file
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"

    # Load the Parquet file into a pandas DataFrame
    df = pd.read_parquet(url, engine='pyarrow')
    print(df.info())

    return df