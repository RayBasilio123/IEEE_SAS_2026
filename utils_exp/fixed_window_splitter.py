from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import AbstractBaseSplitter, slice_data_entry

if TYPE_CHECKING:
    from collections.abc import Generator

    from gluonts.dataset.common import DataEntry, Dataset

logger = logging.getLogger(__file__)


@dataclass
class FixedWindowSplitter(AbstractBaseSplitter):
    """
    A splitter that implements walk-forward validation (also known as time series cross-validation).

    The model is trained using a fixed window of past observations, and testing is performed
    on a rolling basis where the training window is moved forward in time. The size of the
    training window is kept constant, allowing for the model to be tested on different
    sections of the data.

    Parameters
    ----------
    window_size
        The number of past observations to use for training (context length).
    step_size
        The number of observations to move forward for each test window.
        If None, defaults to 1 (one step forward each iteration).
    """

    window_size: int
    step_size: Optional[int] = None

    def training_entry(self, entry: DataEntry) -> DataEntry:
        """
        Training entry is not used in walk-forward validation since each fold
        has its own training window, but we implement it for the abstract interface.
        """
        return slice_data_entry(entry, slice(None, self.window_size))

    def test_pair(
        self, entry: DataEntry, prediction_length: int, offset: int = 0
    ) -> tuple[DataEntry, DataEntry]:
        """
        Generate a test pair with fixed window training data and corresponding labels.

        The input (training) slice has a fixed size of window_size, and the label
        (test) slice has size prediction_length, both starting at position offset.
        """
        # Ensure offset is non-negative
        if offset < 0:
            offset += entry[FieldName.TARGET].shape[-1]

        # Training window: from (offset - window_size) to offset
        train_start = max(0, offset - self.window_size)
        train_end = offset

        # Test window: from offset to (offset + prediction_length)
        test_start = offset
        test_end = offset + prediction_length

        # Validate that we have enough data
        assert (
            test_end <= entry[FieldName.TARGET].shape[-1]
        ), f"Not enough data for test window: test_end={test_end}, target_length={entry[FieldName.TARGET].shape[-1]}"

        input_slice = slice(train_start, train_end)
        label_slice = slice(test_start, test_end)

        return (
            slice_data_entry(
                entry, input_slice, prediction_length=prediction_length
            ),
            slice_data_entry(
                entry, label_slice, prediction_length=prediction_length
            ),
        )

    def generate_test_pairs(
        self,
        dataset: Dataset,
        prediction_length: int,
        windows: int = 1,
        distance: Optional[int] = None,
        max_history: Optional[int] = None,
    ) -> Generator[tuple[DataEntry, DataEntry], None, None]:
        """
        Generate test pairs using walk-forward validation.

        Parameters
        ----------
        dataset
            Dataset to generate test pairs from.
        prediction_length
            Length of the prediction horizon.
        windows
            Number of windows to generate (not used in walk-forward).
        distance
            Step size for rolling window. If None, defaults to step_size or 1.
        max_history
            Maximum history to keep. If specified, training data is truncated.
        """
        if distance is None:
            distance = self.step_size if self.step_size is not None else 1

        splits_num = 0
        for entry in dataset:
            total_length = entry[FieldName.TARGET].shape[-1]
            # Start from window_size position (first valid offset)
            current_offset = self.window_size

            while current_offset + prediction_length <= total_length:
                test = self.test_pair(
                    entry,
                    prediction_length=prediction_length,
                    offset=current_offset,
                )

                # Apply max_history constraint if specified
                if max_history is not None:
                    test_input = slice_data_entry(
                        test[0], slice(-max_history, None)
                    )
                    yield test_input, test[1]
                else:
                    yield test[0], test[1]

                logger.debug(
                    f"Generated test pair - context start {test[0][FieldName.START]}; "
                    f"forecast start {test[1][FieldName.START]} - "
                    f"{test[0].get('item_id', 'unknown')}"
                )
                current_offset += distance
                splits_num += 1

        logger.info(f"Generated {splits_num} test pairs using walk-forward validation")
        if splits_num == 0:
            raise ValueError("No test pairs were generated")
