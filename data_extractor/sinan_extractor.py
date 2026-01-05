from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from gluonts.dataset.arrow import ParquetWriter
from gluonts.dataset.pandas import PandasDataset
from pysus.ftp.databases.sinan import SINAN

if TYPE_CHECKING:
    from pysus.data.local import ParquetSet


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__package__)
ERRORS = []
DEFAULT_PATH = "data"

SELECTED_DISEASES = [
    "BOTU",
    "CHAG",
    "CHIK",
    "COLE",
    "COQU",
    "DERM",
    "DIFT",
    "ESQU",
    "EXAN",
    "FMAC",
    "FTIF",
    "HANS",
    "HANT",
    "HEPA",
    "INFL",
    "LEIV",
    "LEPT",
    "LERD",
    "LTAN",
    "MALA",
    "MENI",
    "NTRA",
    "PEST",
    "PFAN",
    "PNEU",
    "RAIV",
    "SDTA",
    "SIFA",
    "SIFC",
    "SIFG",
    "SRC",
    "TETA",
    "TETN",
    "TOXC",
    "TOXG",
    "TRAC",
    "TUBE",
    "VARC",
    "ZIKA",
#    "DENG",
]


class SinanDataExtractor:
    def __init__(
        self,
        disease_codes: list[str],
        path: str,
        frequency: str,
    ) -> None:
        self.path = path if path else DEFAULT_PATH
        self.disease_codes = disease_codes if disease_codes else SELECTED_DISEASES
        self.frequency = frequency if frequency else "ME"
        self.sinan = SINAN().load()

    def extract(self) -> None:
        logger.info(f"Extracting SINAN data to {self.path}")
        cases: list[ParquetSet] = self.sinan.download(
            self.sinan.get_files(self.disease_codes), f"{self.path}/raw"
        )
        self._preprocessing(cases)
        logger.info(f"Finished extracting SINAN data to {self.path}")
        logger.error(f"Errors: {ERRORS}")

    def _get_disease_code_by_filename(self, case: ParquetSet) -> str:
        filename = case.path.split("/")[-1]
        filename_without_extension = filename.split(".")[0]
        disease_code = "".join(filter(str.isalpha, filename_without_extension))
        assert disease_code is not None
        return disease_code

    def _get_dataset_year_by_filename(self, case: ParquetSet) -> str:
        filename = case.path.split("/")[-1]
        filename_without_extension = filename.split(".")[0]
        return "".join(filter(str.isdigit, filename_without_extension))

    def _preprocessing(self, cases: list[ParquetSet]) -> None:
        files_by_disease = defaultdict(list)

        for case in cases:
            disease_code = self._get_disease_code_by_filename(case)
            files_by_disease[disease_code].append(case)

        for disease_name in files_by_disease.keys():
            try:
                df_merged = pd.DataFrame()
                for case in files_by_disease[disease_name]:
                    dataset_year = self._get_dataset_year_by_filename(case)
                    logger.info(
                        f"Processing disease dataset: {disease_name} - {dataset_year}"
                    )
                    try:
                        df: pd.DataFrame = case.to_dataframe()

                        # Getting State code as the key
                        df["item_id"] = pd.to_numeric(
                            (df["ID_MUNICIP"].astype(str, errors="raise").str[:2]),
                            downcast="unsigned",
                        )
                        df = self.remove_inconsistent_dates(df, dataset_year)
                        df = df[["item_id", "timestamp"]]  # type: ignore
                        df_merged = pd.concat([df_merged, df], ignore_index=True)
                    except Exception as e:
                        logger.error(
                            f"Failed to load dataset: {disease_name} - {dataset_year}",
                            exc_info=True,
                        )

                # Group by unique_id and ds and sum the cases
                grouped_df = (
                    df_merged.groupby(
                        [
                            pd.Grouper(key="timestamp", freq=self.frequency),
                            "item_id",
                        ]
                    )
                    .size()
                    .reset_index(name="target")
                )

                grouped_df = self._fill_missing_date(grouped_df).sort_values(
                    by=["item_id", "timestamp"]
                )

                # Check if the two dates are too far apart
                max_date = grouped_df["timestamp"].max()
                min_date = grouped_df["timestamp"].min()
                logger.info(
                    f"Max date: {max_date}; Min date: {min_date} of {disease_name}"
                )

                if (max_date.year - min_date.year) > 25:
                    raise ValueError(
                        "There are more than 24 years between the max and min date in the dataset, it should be a problem"
                    )
                
                grouped_df = self.filter_by_threshold(grouped_df)
                SAVE_PATH = f"{self.path}/processed_gluonts"
                Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
                self.to_gluonts_format(
                    grouped_df, f"{SAVE_PATH}/{disease_name}.parquet"
                )
                logger.info(f"Saved: {disease_name} in {SAVE_PATH}")
            except Exception as e:
                logger.error(
                    f"Failed to process disease dataset: {disease_name}", exc_info=True
                )
                ERRORS.append(disease_name)
                continue
    def filter_by_threshold(self, df: pd.DataFrame, length_threshold: int = 80, mean_threshold: float = 50) -> pd.DataFrame:
        """Filters out items with a length below a specified threshold and a mean target value below a specified threshold.

        This method removes items from the dataset whose length is below the given threshold and whose mean target value is below the given threshold. This can help in focusing on items with more significant activity.

        Args:
            df: DataFrame with the dataset.
            length_threshold: The length threshold below which items will be removed.
            mean_threshold: The mean target value threshold below which items will be removed.

        Returns:
            DataFrame with items filtered by the length and mean target value thresholds.
        """
        item_lengths = df.groupby("item_id").size()
        mean_values = df.groupby("item_id")["target"].mean()
        filtered_items = mean_values[(item_lengths > length_threshold) & (mean_values > mean_threshold)].index
        
        if len(filtered_items) < len(mean_values):
            removed_items = mean_values[(item_lengths <= length_threshold) | (mean_values <= mean_threshold)].index
            logger.warning(
                f"Removing items with length below {length_threshold} or mean target value below {mean_threshold}: {removed_items.tolist()}"
            )
        
        filtered_df = df[df["item_id"].isin(filtered_items)]
        
        if filtered_df.empty:
            raise ValueError("The resulting DataFrame is empty after filtering by length and mean thresholds.")
        
        if len(filtered_df["item_id"].unique()) < 5:
            raise ValueError("The number of unique item IDs is less than 5, which is insufficient for analysis.")
        
        return filtered_df
    def _fill_missing_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing dates with 0 cases in the provided DataFrame based on the selected frequency.

        This method ensures that all dates within the range of the dataset are represented according to the specified frequency,
        even if there are no notifications for some periods. This is important because the absence of notifications in a period
        means that period does not appear in the dataset and there are no notifications in SINAN. While this approach does not
        address the issue of underreporting, it ensures consistency in the time series data by not artificially inflating or
        deflating the number of cases. The subnotification can be a problem, but it may occur even in periods with notifications,
        so, in the same way we do not increase the number of cases, we do not decrease it.
        """
        all_dates = pd.date_range(
            start=df["timestamp"].min(),
            end=df["timestamp"].max(),
            freq=self.frequency,
        )
        all_items = df["item_id"].unique()

        idx = pd.MultiIndex.from_product(
            [all_dates, all_items],
            names=["timestamp", "item_id"],
        )

        return (
            df.set_index(["timestamp", "item_id"])
            .reindex(idx, fill_value=0)
            .reset_index()
        )

    def remove_inconsistent_dates(
        self, df: pd.DataFrame, dataset_year: str
    ) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
        failed_notif = df["DT_NOTIFIC"].isna() & ~df["DT_NOTIFIC"].isna()
        if failed_notif.any():
            logger.warning(
                f"Failed to convert {failed_notif.sum()} notification dates:"
            )
            logger.warning(df["DT_NOTIFIC"][failed_notif].value_counts())
        year_threshold = 1998
        early_dates = df[df["timestamp"].dt.year < year_threshold]
        if not early_dates.empty:
            logger.warning(
                f"Found {len(early_dates)} rows with dates before {year_threshold}:"
            )
            logger.warning(early_dates[["DT_NOTIFIC"]])
            df = df[df["timestamp"].dt.year >= year_threshold]  # type: ignore
        # Log and drop invalid rows in one operation
        if df["timestamp"].isna().any():
            logger.warning(f"Dropping rows with invalid dates")
            df = df.dropna(subset=["timestamp"])
        return df

    def to_gluonts_format(self, df: pd.DataFrame, path: str) -> None:
        df.set_index(["timestamp"], inplace=True)
        gluonts_df = PandasDataset.from_long_dataframe(df, item_id="item_id", freq="M")
        ParquetWriter(metadata={"freq": "M"}).write_to_file(gluonts_df, path)


if __name__ == "__main__":
    extractor = SinanDataExtractor(
        disease_codes=SELECTED_DISEASES,
        path=DEFAULT_PATH,
        frequency="ME",
        split_rate=None,
    )
    extractor.extract()
