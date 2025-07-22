"""
GSC Data Processor with Flexible Column Mapping
Enterprise-grade CSV processing for GSC Performance data
"""
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ColumnMappingError(Exception):
    """Raised when required columns cannot be mapped from CSV."""
    pass


class GSCColumnType(Enum):
    """Enumeration of GSC column types for mapping."""
    QUERY = "query"
    URL = "url" 
    CLICKS = "clicks"
    IMPRESSIONS = "impressions"
    CTR = "ctr"
    POSITION = "position"


@dataclass
class ColumnMappingConfig:
    """Configuration for column name variations and their mappings."""
    
    # Column name variations (case-insensitive)
    COLUMN_VARIATIONS: Dict[GSCColumnType, List[str]] = None
    
    def __post_init__(self):
        """Initialize column variations if not provided."""
        if self.COLUMN_VARIATIONS is None:
            self.COLUMN_VARIATIONS = {
                GSCColumnType.QUERY: [
                    "query", "queries", "keyword", "keywords", "keyqords",
                    "search term", "search terms", "search query", "search queries"
                ],
                GSCColumnType.URL: [
                    "url", "urls", "page", "pages", "landing page", "landing pages",
                    "address", "addresses", "destination url", "destination page"
                ],
                GSCColumnType.CLICKS: [
                    "click", "clicks", "total clicks", "click count"
                ],
                GSCColumnType.IMPRESSIONS: [
                    "impression", "impressions", "total impressions", "impression count"
                ],
                GSCColumnType.CTR: [
                    "ctr", "url ctr", "click through rate", "click-through rate",
                    "clickthrough rate", "click thru rate"
                ],
                GSCColumnType.POSITION: [
                    "position", "positions", "avg pos", "avg. pos", "avg position",
                    "average position", "avg. position", "ranking", "rank"
                ]
            }


class GSCDataProcessor:
    """
    Enterprise GSC data processor with flexible column mapping.
    
    Handles various GSC export formats and column naming conventions
    while maintaining data integrity and providing comprehensive error handling.
    """
    
    def __init__(self, config: Optional[ColumnMappingConfig] = None):
        """
        Initialize GSC processor with column mapping configuration.
        
        Parameters
        ----------
        config : ColumnMappingConfig, optional
            Configuration for column name variations, by default None
        """
        self.config = config or ColumnMappingConfig()
        self._column_mapping: Dict[str, str] = {}
        
    def process_gsc_data(
        self, 
        csv_file_path: str,
        required_columns: Optional[List[GSCColumnType]] = None
    ) -> pd.DataFrame:
        """
        Process GSC CSV data with flexible column mapping.
        
        Parameters
        ----------
        csv_file_path : str
            Path to the GSC CSV file
        required_columns : List[GSCColumnType], optional
            Required columns for processing, by default [QUERY, URL, CLICKS]
            
        Returns
        -------
        pd.DataFrame
            Processed dataframe with standardized column names
            
        Raises
        ------
        ColumnMappingError
            When required columns cannot be mapped
        FileNotFoundError
            When CSV file cannot be found
        """
        # Set default required columns
        if required_columns is None:
            required_columns = [
                GSCColumnType.QUERY, 
                GSCColumnType.URL, 
                GSCColumnType.CLICKS
            ]
            
        logger.info(
            "Processing GSC data",
            file_path=csv_file_path,
            required_columns=[col.value for col in required_columns]
        )
        
        try:
            # Load CSV data
            df = self._load_csv_data(csv_file_path)
            
            # Create column mapping
            column_mapping = self._create_column_mapping(df.columns, required_columns)
            
            # Apply column mapping
            df_mapped = self._apply_column_mapping(df, column_mapping)
            
            # Validate and clean data
            df_clean = self._validate_and_clean_data(df_mapped, required_columns)
            
            logger.info(
                "GSC data processed successfully",
                rows_processed=len(df_clean),
                columns_mapped=list(column_mapping.keys())
            )
            
            return df_clean
            
        except Exception as e:
            logger.error(
                "Failed to process GSC data", 
                file_path=csv_file_path,
                error=str(e)
            )
            raise
    
    def _load_csv_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data with encoding detection and error handling."""
        try:
            # Try common encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.debug(f"Successfully loaded CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
                    
            # If all encodings fail, raise error
            raise ValueError(f"Could not decode CSV file with any common encoding")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"GSC CSV file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def _create_column_mapping(
        self, 
        csv_columns: List[str], 
        required_columns: List[GSCColumnType]
    ) -> Dict[str, str]:
        """
        Create mapping from CSV columns to standardized column names.
        
        Parameters
        ----------
        csv_columns : List[str]
            Column names from the CSV file
        required_columns : List[GSCColumnType]
            Required column types for the analysis
            
        Returns
        -------
        Dict[str, str]
            Mapping from original column names to standardized names
            
        Raises
        ------
        ColumnMappingError
            When required columns cannot be mapped
        """
        column_mapping = {}
        mapped_types = set()
        
        # Normalize CSV column names for comparison
        normalized_csv_columns = {
            col.lower().strip(): col for col in csv_columns
        }
        
        logger.debug(
            "Creating column mapping",
            csv_columns=list(csv_columns),
            normalized_columns=list(normalized_csv_columns.keys())
        )
        
        # Try to match each column type
        for col_type in GSCColumnType:
            variations = self.config.COLUMN_VARIATIONS[col_type]
            matched_column = None
            
            # Try each variation
            for variation in variations:
                normalized_variation = variation.lower().strip()
                
                # Direct match
                if normalized_variation in normalized_csv_columns:
                    matched_column = normalized_csv_columns[normalized_variation]
                    break
                
                # Partial match (contains)
                for norm_csv_col, original_csv_col in normalized_csv_columns.items():
                    if normalized_variation in norm_csv_col or norm_csv_col in normalized_variation:
                        matched_column = original_csv_col
                        break
                
                if matched_column:
                    break
            
            # If matched, add to mapping
            if matched_column:
                column_mapping[matched_column] = col_type.value
                mapped_types.add(col_type)
                logger.debug(
                    f"Mapped column: {matched_column} -> {col_type.value}"
                )
        
        # Check if all required columns are mapped
        missing_required = set(required_columns) - mapped_types
        if missing_required:
            missing_names = [col.value for col in missing_required]
            available_columns = list(csv_columns)
            
            error_msg = (
                f"Missing required columns: {', '.join(missing_names)}. "
                f"Available columns: {', '.join(available_columns)}. "
                f"Please ensure your CSV contains columns for: {', '.join(missing_names)}"
            )
            
            logger.error(
                "Column mapping failed",
                missing_required=missing_names,
                available_columns=available_columns,
                mapped_columns=list(column_mapping.keys())
            )
            
            raise ColumnMappingError(error_msg)
        
        return column_mapping
    
    def _apply_column_mapping(
        self, 
        df: pd.DataFrame, 
        column_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """Apply column mapping to dataframe."""
        # Create rename mapping
        rename_mapping = column_mapping
        
        # Select and rename only mapped columns
        mapped_df = df[list(rename_mapping.keys())].rename(columns=rename_mapping)
        
        logger.debug(
            "Applied column mapping",
            original_columns=list(df.columns),
            mapped_columns=list(mapped_df.columns)
        )
        
        return mapped_df
    
    def _validate_and_clean_data(
        self, 
        df: pd.DataFrame, 
        required_columns: List[GSCColumnType]
    ) -> pd.DataFrame:
        """
        Validate and clean the mapped dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with mapped columns
        required_columns : List[GSCColumnType]
            Required column types
            
        Returns
        -------
        pd.DataFrame
            Cleaned and validated dataframe
        """
        df_clean = df.copy()
        
        # Remove rows where required columns are null
        required_col_names = [col.value for col in required_columns]
        initial_rows = len(df_clean)
        
        df_clean = df_clean.dropna(subset=required_col_names)
        
        rows_removed = initial_rows - len(df_clean)
        if rows_removed > 0:
            logger.warning(
                f"Removed {rows_removed} rows with missing required data",
                required_columns=required_col_names
            )
        
        # Clean and validate numeric columns
        numeric_columns = ['clicks', 'impressions', 'ctr', 'position']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows with 0 clicks (as per requirements)
        if 'clicks' in df_clean.columns:
            initial_rows_clicks = len(df_clean)
            df_clean = df_clean[df_clean['clicks'] > 0]
            
            zero_click_rows_removed = initial_rows_clicks - len(df_clean)
            if zero_click_rows_removed > 0:
                logger.info(
                    f"Removed {zero_click_rows_removed} rows with 0 clicks",
                    remaining_rows=len(df_clean)
                )
        
        # Clean URL and query columns
        text_columns = ['url', 'query']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # Remove empty strings
                df_clean = df_clean[df_clean[col] != '']
        
        logger.info(
            "Data validation completed",
            final_rows=len(df_clean),
            columns=list(df_clean.columns)
        )
        
        return df_clean
    
    def get_column_mapping_info(self) -> Dict[str, Any]:
        """Get information about available column mappings."""
        return {
            "supported_variations": {
                col_type.value: variations 
                for col_type, variations in self.config.COLUMN_VARIATIONS.items()
            },
            "required_columns": ["query", "url", "clicks"],
            "optional_columns": ["impressions", "ctr", "position"]
        }


def validate_gsc_file(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate a GSC CSV file and return mapping information.
    
    Parameters
    ----------
    file_path : str
        Path to the GSC CSV file
        
    Returns
    -------
    Tuple[bool, str, Dict[str, Any]]
        (is_valid, message, column_info)
    """
    try:
        processor = GSCDataProcessor()
        
        # Try to process with minimal requirements
        df = processor.process_gsc_data(
            file_path, 
            required_columns=[GSCColumnType.QUERY, GSCColumnType.URL, GSCColumnType.CLICKS]
        )
        
        column_info = {
            "total_rows": len(df),
            "available_columns": list(df.columns),
            "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
        }
        
        return True, "File successfully validated", column_info
        
    except Exception as e:
        return False, str(e), {}
