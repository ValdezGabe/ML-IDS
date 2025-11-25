"""
Data loader for UNSW-NB15 dataset
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
from typing import Tuple, Optional
import logging

from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNSWDataLoader:
    """Load and prepare UNSW-NB15 dataset"""

    # UNSW-NB15 dataset URLs
    DATASET_URLS = {
        'training': 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_1.csv',
        'testing': 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_2.csv',
        'features': 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=NUSW-NB15_features.csv'
    }

    # Column names for UNSW-NB15
    COLUMN_NAMES = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
        'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
        'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
        'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
        'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
        'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst',
        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
        'ct_dst_src_ltm', 'attack_cat', 'label'
    ]

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader

        Args:
            data_dir: Directory to store/load data from
        """
        self.data_dir = data_dir or Config.RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download UNSW-NB15 dataset

        Args:
            force_download: Force re-download even if files exist

        Returns:
            True if successful, False otherwise
        """
        logger.info("Downloading UNSW-NB15 dataset...")

        # Note: The official UNSW-NB15 URLs may require manual download
        # Users should manually download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset

        logger.warning(
            "Please manually download the UNSW-NB15 dataset from:\n"
            "https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
            "Download the following CSV files:\n"
            "  - UNSW-NB15_1.csv (training set)\n"
            "  - UNSW-NB15_2.csv (testing set)\n"
            "  - UNSW-NB15_3.csv (additional data)\n"
            "  - UNSW-NB15_4.csv (additional data)\n"
            f"Place them in: {self.data_dir}\n"
        )

        return False

    def load_data(self,
                  dataset_type: str = 'training',
                  sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load UNSW-NB15 dataset

        Args:
            dataset_type: 'training', 'testing', or 'full'
            sample_size: Optional sample size for quick testing

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading {dataset_type} dataset...")

        # Map dataset types to files
        file_mapping = {
            'training': ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv'],
            'testing': ['UNSW-NB15_3.csv'],
            'full': ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
        }

        files_to_load = file_mapping.get(dataset_type, file_mapping['training'])

        dfs = []
        for filename in files_to_load:
            filepath = self.data_dir / filename

            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                logger.info("Please download the dataset manually (see download_dataset() for instructions)")
                continue

            try:
                df = pd.read_csv(filepath, header=None, names=self.COLUMN_NAMES)
                dfs.append(df)
                logger.info(f"Loaded {filename}: {len(df)} records")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

        if not dfs:
            raise FileNotFoundError(
                f"No dataset files found in {self.data_dir}. "
                "Please download the UNSW-NB15 dataset manually."
            )

        # Combine all dataframes
        df = pd.concat(dfs, ignore_index=True)

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records")

        logger.info(f"Total records loaded: {len(df)}")

        return df

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get information about the dataset

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'total_records': len(df),
            'features': len(df.columns),
            'attack_distribution': df['attack_cat'].value_counts().to_dict(),
            'label_distribution': df['label'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }

        return info

    def create_train_test_split(self,
                                df: pd.DataFrame,
                                test_size: float = 0.2,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets

        Args:
            df: DataFrame to split
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        # Stratify by attack category to maintain class distribution
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['attack_cat'],
            random_state=random_state
        )

        logger.info(f"Training set: {len(train_df)} records")
        logger.info(f"Testing set: {len(test_df)} records")

        return train_df, test_df


def load_unsw_nb15(dataset_type: str = 'training',
                   sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function to load UNSW-NB15 dataset

    Args:
        dataset_type: 'training', 'testing', or 'full'
        sample_size: Optional sample size for quick testing

    Returns:
        DataFrame with loaded data
    """
    loader = UNSWDataLoader()
    return loader.load_data(dataset_type, sample_size)


if __name__ == "__main__":
    # Example usage
    loader = UNSWDataLoader()

    # Show download instructions
    loader.download_dataset()

    # Try to load data (will fail if not downloaded)
    try:
        df = loader.load_data('training', sample_size=1000)
        info = loader.get_data_info(df)

        print("\nDataset Info:")
        print(f"Total records: {info['total_records']}")
        print(f"Features: {info['features']}")
        print(f"\nAttack distribution:")
        for attack, count in info['attack_distribution'].items():
            print(f"  {attack}: {count}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
