from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import AbstractBaseSplitter, slice_data_entry

if TYPE_CHECKING:
    from collections.abc import Generator

    from gluonts.dataset.common import DataEntry, Dataset

logger = logging.getLogger(__file__)


@dataclass
class TSFMExperimentSplitter(AbstractBaseSplitter):
    """
    A splitter that slices data to evaluate the model's performance in a time series forecasting setting.

    Instead of using fixed windows to split the data, maximize the splits based on the context length while accounting for the prediction length.
    Ensure that the available time points are sufficient both for the model's input and for evaluating the predictions.

    Parameters
    ----------
    offset
        Offset determining where the training data ends.
        A positive offset indicates how many observations since the start of
        each series should be in the training slice; a negative offset
        indicates how many observations before the end of each series should
        be excluded from the training slice.
    """

    offset: int = 0
    context_length: int = 32

    def training_entry(self, entry: DataEntry) -> DataEntry:
        # It isn't necessary to the experiment, but it need to be implemented to avoid errors
        return slice_data_entry(entry, slice(None, self.offset))

    def test_pair(
        self, entry: DataEntry, prediction_length: int, offset: int = 0
    ) -> tuple[DataEntry, DataEntry]:
        if self.offset < 0:
            offset += entry[FieldName.TARGET].shape[-1]

        input_slice = slice(max(0, offset - self.context_length), offset)
        label_slice = slice(offset, offset + prediction_length)

        return (
            slice_data_entry(entry, input_slice, prediction_length=prediction_length),
            slice_data_entry(entry, label_slice, prediction_length=prediction_length),
        )

    def generate_test_pairs(
        self,
        dataset: Dataset,
        prediction_length: int,
        windows: int = 1,
        distance: int | None = None,
        max_history: int | None = None,
    ) -> Generator[tuple[DataEntry, DataEntry], None, None]:
        """
        Generate test pairs from a dataset.

        Parameters
        ----------
        dataset
            Dataset to generate test pairs from.
        prediction_length
            Prediction length.
        windows
            This parameter is not used in this implementation.
        distance
            Distance between windows. Thw default is the context length.
        max_history
            This parameter is not used in this implementation.
        """
        if distance is None:
            distance = self.context_length
        splits_num = 0
        for entry in dataset:
            total_length = entry[FieldName.TARGET].shape[-1]
            current_offset = self.context_length
            while current_offset + prediction_length <= total_length:
                test = self.test_pair(
                    entry, prediction_length=prediction_length, offset=current_offset
                )
                logger.debug(f'Generated test pair - context start {test[0]["start"]}; forecast start {test[1]["start"]} - {test[0]["item_id"]}')
                yield test[0], test[1]
                current_offset += distance
                splits_num += 1
        logger.info(f"Generated {splits_num} test pairs")
        if splits_num == 0:
            raise ValueError("No test pairs were generated")
