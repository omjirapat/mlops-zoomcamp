from batch_q3 import prepare_data
import pandas as pd
from datetime import datetime
columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def create_data() -> pd.DataFrame:
    
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

def expected_data(column_test) -> pd.DataFrame:
    data_expected = [
        ('-1', '-1', 9.0),
        ('1', '1', 8.0),
    ]
    df_expected = pd.DataFrame(data_expected, columns = column_test)
    return df_expected

def test_prepare_data():
    categorical = ['PULocationID', 'DOLocationID']
    df = create_data()
    df_actual = prepare_data(df, categorical)
    columns_test = ['PULocationID', 'DOLocationID', 'duration']
    df_expected = expected_data(columns_test)

    assert(df_actual['PULocationID'] == df_expected['PULocationID']).all()
    assert(df_actual['DOLocationID'] == df_expected['DOLocationID']).all()
    assert(df_actual['duration'] - df_expected['duration']).abs().sum() < 0.000001