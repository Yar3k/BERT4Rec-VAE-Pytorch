from .base import AbstractDataset

import pandas as pd

from datetime import date


class STEAMDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'steam'

    @classmethod
    def url(cls):
        pass

    @classmethod
    def zip_file_content_is_folder(cls):
        pass

    @classmethod
    def all_raw_file_names(cls):
        return ['steam.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('steam.csv')
        df = pd.read_csv(file_path)
        df['rating'] = 5
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


