"""
Hyper File Converter for ML-IDS
Converts CSV files to Tableau .hyper format for direct publishing
"""
from tableauhyperapi import HyperProcess, Telemetry, Connection, CreateMode, \
    NOT_NULLABLE, NULLABLE, SqlType, TableDefinition, Inserter, escape_name, escape_string_literal, TableName
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class HyperConverter:
    """Convert DataFrames and CSV files to Tableau .hyper format"""

    def __init__(self):
        """Initialize Hyper converter"""
        pass

    def dataframe_to_hyper(
        self,
        df: pd.DataFrame,
        hyper_path: Path,
        table_name: str = "Extract"
    ) -> Path:
        """
        Convert pandas DataFrame to Tableau .hyper file

        Args:
            df: DataFrame to convert
            hyper_path: Output path for .hyper file
            table_name: Name of the table in the hyper file

        Returns:
            Path to created .hyper file
        """
        logger.info(f"Converting DataFrame to Hyper file: {hyper_path}")

        # Ensure path is absolute
        hyper_path = Path(hyper_path).absolute()
        hyper_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file if it exists
        if hyper_path.exists():
            hyper_path.unlink()

        # Start Hyper process
        with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
            with Connection(endpoint=hyper.endpoint,
                          database=str(hyper_path),
                          create_mode=CreateMode.CREATE_AND_REPLACE) as connection:

                # Create table definition
                table_def = self._create_table_definition(df, table_name)

                # Create table
                connection.catalog.create_table(table_def)

                # Insert data
                with Inserter(connection, table_def) as inserter:
                    # Convert DataFrame to list of rows
                    for _, row in df.iterrows():
                        inserter.add_row(self._convert_row(row, df.dtypes))
                    inserter.execute()

                logger.info(f"Successfully created Hyper file with {len(df)} rows")

        return hyper_path

    def csv_to_hyper(
        self,
        csv_path: Path,
        hyper_path: Optional[Path] = None,
        table_name: Optional[str] = None
    ) -> Path:
        """
        Convert CSV file to Tableau .hyper file

        Args:
            csv_path: Path to CSV file
            hyper_path: Output path (defaults to same name with .hyper extension)
            table_name: Table name (defaults to CSV filename)

        Returns:
            Path to created .hyper file
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Default hyper path
        if hyper_path is None:
            hyper_path = csv_path.with_suffix('.hyper')

        # Default table name
        if table_name is None:
            table_name = csv_path.stem.replace('_', ' ').title()

        logger.info(f"Converting CSV to Hyper: {csv_path} -> {hyper_path}")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Convert to hyper
        return self.dataframe_to_hyper(df, hyper_path, table_name)

    def batch_csv_to_hyper(
        self,
        csv_files: List[Path],
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Convert multiple CSV files to .hyper format

        Args:
            csv_files: List of CSV file paths
            output_dir: Output directory (defaults to same as CSV)

        Returns:
            List of created .hyper file paths
        """
        hyper_files = []

        for csv_file in csv_files:
            try:
                if output_dir:
                    hyper_path = output_dir / csv_file.with_suffix('.hyper').name
                else:
                    hyper_path = csv_file.with_suffix('.hyper')

                hyper_file = self.csv_to_hyper(csv_file, hyper_path)
                hyper_files.append(hyper_file)
                logger.info(f"✅ Converted: {csv_file.name} -> {hyper_file.name}")

            except Exception as e:
                logger.error(f"Failed to convert {csv_file}: {e}")

        logger.info(f"Converted {len(hyper_files)}/{len(csv_files)} files to Hyper format")
        return hyper_files

    def _create_table_definition(
        self,
        df: pd.DataFrame,
        table_name: str
    ) -> TableDefinition:
        """
        Create Tableau table definition from DataFrame

        Args:
            df: DataFrame
            table_name: Table name

        Returns:
            TableDefinition
        """
        columns = []

        for col_name, dtype in df.dtypes.items():
            # Map pandas dtype to Tableau SQL type
            sql_type = self._pandas_to_sql_type(dtype)

            # Check if column has nulls
            nullability = NULLABLE if df[col_name].isnull().any() else NOT_NULLABLE

            columns.append(
                TableDefinition.Column(col_name, sql_type, nullability)
            )

        # Use simple table name (Tableau will place it in "Extract" schema automatically)
        table_def = TableDefinition(
            table_name=table_name,
            columns=columns
        )

        return table_def

    @staticmethod
    def _pandas_to_sql_type(dtype) -> SqlType:
        """
        Map pandas dtype to Tableau SQL type

        Args:
            dtype: Pandas dtype

        Returns:
            SqlType
        """
        dtype_str = str(dtype)

        # Integer types
        if 'int' in dtype_str:
            if 'int8' in dtype_str or 'int16' in dtype_str:
                return SqlType.small_int()
            elif 'int32' in dtype_str:
                return SqlType.int()
            else:
                return SqlType.big_int()

        # Float types
        elif 'float' in dtype_str:
            return SqlType.double()

        # Boolean
        elif dtype_str == 'bool':
            return SqlType.bool()

        # Datetime
        elif 'datetime' in dtype_str:
            return SqlType.timestamp()

        # Date
        elif 'date' in dtype_str:
            return SqlType.date()

        # Default to text
        else:
            return SqlType.text()

    @staticmethod
    def _convert_row(row: pd.Series, dtypes: pd.Series) -> List:
        """
        Convert pandas row to list with proper type handling

        Args:
            row: Pandas Series (row)
            dtypes: Series of dtypes

        Returns:
            List of values
        """
        values = []

        for col_name, value in row.items():
            # Handle NaN/None
            if pd.isna(value):
                values.append(None)
            # Handle datetime
            elif 'datetime' in str(dtypes[col_name]):
                values.append(pd.Timestamp(value))
            # Handle boolean
            elif dtypes[col_name] == 'bool':
                values.append(bool(value))
            # Handle numeric
            elif 'int' in str(dtypes[col_name]):
                values.append(int(value))
            elif 'float' in str(dtypes[col_name]):
                values.append(float(value))
            # Default to string
            else:
                values.append(str(value))

        return values
