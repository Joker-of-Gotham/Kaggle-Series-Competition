"""Default gateway notebook"""

import os
from pathlib import Path

import pandas as pd
import polars as pl

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.templates


class DefaultGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths, file_share_dir=None)
        self.data_paths = data_paths
        self.row_id_column_name = 'batch_id'
        self.target_column_name = 'prediction'
        self.set_response_timeout_seconds(60 * 5)

    def unpack_data_paths(self):
        if self.data_paths:
            self.competition_data_dir = self.data_paths[0]
        else:
            self.competition_data_dir = Path(__file__).parent.parent
        self.competition_data_dir = Path(self.competition_data_dir)

    def generate_data_batches(self):
        test = pl.read_csv(self.competition_data_dir / 'test.csv')

        if self.row_id_column_name not in test.columns:
            self.row_id_column_name = test.columns[0]
            assert test[self.row_id_column_name].is_sorted()

        batch_ids = test[self.row_id_column_name].unique(maintain_order=True).to_list()
        for batch_id in batch_ids:
            test_batch = test.filter(pl.col(self.row_id_column_name) == batch_id)
            yield (
                (test_batch,),
                batch_id,
            )

    def competition_specific_validation(self, prediction, row_ids, data_batch) -> None:
        pass


if __name__ == '__main__':
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        gateway = DefaultGateway()
        # Relies on valid default data paths
        gateway.run()
    else:
        print('Skipping run for now')
