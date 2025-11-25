"""
Script to download and prepare UNSW-NB15 dataset
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import UNSWDataLoader
from src.utils.config import Config


def main():
    """Main function to guide dataset download"""

    print("=" * 80)
    print("UNSW-NB15 Dataset Download Guide")
    print("=" * 80)
    print()

    loader = UNSWDataLoader()

    print("Step 1: Visit the UNSW-NB15 dataset page")
    print("URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print()

    print("Step 2: Download the following CSV files:")
    print("  - UNSW-NB15_1.csv")
    print("  - UNSW-NB15_2.csv")
    print("  - UNSW-NB15_3.csv")
    print("  - UNSW-NB15_4.csv")
    print()

    print(f"Step 3: Place the downloaded files in:")
    print(f"  {loader.data_dir}")
    print()

    # Create directory if it doesn't exist
    loader.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {loader.data_dir}")
    print()

    # Check if files exist
    required_files = [
        'UNSW-NB15_1.csv',
        'UNSW-NB15_2.csv',
        'UNSW-NB15_3.csv',
        'UNSW-NB15_4.csv'
    ]

    print("Checking for existing files:")
    files_present = []
    for filename in required_files:
        filepath = loader.data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
            files_present.append(filename)
        else:
            print(f"  ✗ {filename} (not found)")

    print()

    if len(files_present) > 0:
        print(f"Found {len(files_present)}/{len(required_files)} files")
        print()

        # Test loading
        response = input("Would you like to test loading the dataset? (y/n): ")
        if response.lower() == 'y':
            try:
                print("\nLoading sample data (1000 records)...")
                df = loader.load_data('training', sample_size=1000)

                print("\nDataset loaded successfully!")
                print(f"Shape: {df.shape}")
                print(f"\nFirst few rows:")
                print(df.head())

                print("\nAttack distribution:")
                print(df['attack_cat'].value_counts())

            except Exception as e:
                print(f"\nError loading dataset: {e}")
    else:
        print("No dataset files found. Please download them from the URL above.")

    print()
    print("=" * 80)
    print("Note: You can also use Kaggle to download the dataset:")
    print("https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
    print("=" * 80)


if __name__ == "__main__":
    main()
