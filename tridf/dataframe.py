from typing import List

import torch


class DataFrame:
    def __init__(self, data: List[torch.Tensor], columns: List[str] = None):
        assert isinstance(data, List)

        if columns is None:
            columns = list(range(len(data)))

        assert len(columns) == len(data)
        self._name_to_column = dict()
        for col_name, col in zip(columns, data):
            self._name_to_column[col_name] = col
