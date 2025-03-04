import pathlib
import pandas as pd

def getDataFromDataset():
    file_path = pathlib.Path(__file__).cwd() / 'src' / 'dataset'

    db_df = pd.read_csv(file_path / 'district_blocks.csv')
    es_df = pd.read_csv(file_path / 'evacuation_spaces.csv')

    db_coordinates = list(zip(db_df['Longitude'], db_df['Latitude']))
    es_coordinates = list(zip(es_df['Longitude'], es_df['Latitude']))

    return db_coordinates, es_coordinates
