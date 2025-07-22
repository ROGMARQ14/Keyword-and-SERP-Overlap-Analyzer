"""
Enhanced GSC Data Processor with Automatic Delimiter Detection
Production-grade CSV processing for enterprise environments
"""

import pandas as pd
import csv
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import chardet
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class CSVDialectDetector:
    """Enterprise-grade CSV dialect detection with fallback mechanisms."""
    
    def __init__(self):
        self.common_delimiters = [',', ';', '\t', '|', ':']
        self.sample_size = 8192  # Bytes to sample for detection
    
    def detect_delimiter(self, file_path: str) -> str:
        """
        Detect CSV delimiter using multiple methods with fallbacks.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Detected delimiter character
        """
        try:
            # Method 1: Use csv.Sniffer with restricted delimiters
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(self.sample_size)
                
            if sample:
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample, delimiters=''.join(self.common_delimiters))
                    logger.info(f"CSV Sniffer detected delimiter: '{dialect.delimiter}'")
                    return dialect.delimiter
                except csv.Error:
                    logger.warning("CSV Sniffer failed, trying frequency analysis")
            
            # Method 2: Frequency analysis fallback
            delimiter = self._frequency_based_detection(sample)
            if delimiter:
                logger.info(f"Frequency analysis detected delimiter: '{delimiter}'")
                return delimiter
                
            # Method 3: Line consistency check
            delimiter = self._line_consistency_check(file_path)
            if delimiter:
                logger.info(f"Line consistency check detected delimiter: '{delimiter}'")
                return delimiter
                
        except Exception as e:
            logger.error(f"Delimiter detection error: {e}")
        
        # Default fallback
        logger.warning("Using default comma delimiter")
        return ','
    
    def _frequency_based_detection(self, sample: str) -> Optional[str]:
        """Detect delimiter based on character frequency in first few lines."""
        if not sample:
            return None
            
        lines = sample.split('\n')[:5]  # Check first 5 lines
        delimiter_counts = {delim: 0 for delim in self.common_delimiters}
        
        for line in lines:
            if line.strip():
                for delim in self.common_delimiters:
                    delimiter_counts[delim] += line.count(delim)
        
        # Find delimiter with highest consistent count
        max_count = max(delimiter_counts.values())
        if max_count > 0:
            for delim, count in delimiter_counts.items():
                if count == max_count:
                    return delim
        
        return None
    
    def _line_consistency_check(self, file_path: str) -> Optional[str]:
        """Check delimiter consistency across multiple lines."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline().strip() for _ in range(10) if f.readable()]
            
            for delimiter in self.common_delimiters:
                field_counts = [line.count(delimiter) for line in lines if line]
                if field_counts and len(set(field_counts)) == 1 and field_counts[0] > 0:
                    return delimiter
                    
        except Exception as e:
            logger.error(f"Line consistency check failed: {e}")
        
        return None


class EnhancedGSCProcessor:
    """
    Enterprise-grade GSC data processor with automatic delimiter detection
    and flexible column mapping.
    """
    
    def __init__(self, config: Optional[ColumnMappingConfig] = None):
        """Initialize processor with configuration."""
        self.config = config or ColumnMappingConfig()
        self.dialect_detector = CSVDialectDetector()
    
    def process_gsc_data(
        self, 
        file_path: str,
        required_columns: Optional[List[GSCColumnType]] = None
    ) -> pd.DataFrame:
        """
        Process GSC CSV data with automatic delimiter detection and column mapping.
        
        Args:
            file_path: Path to the GSC CSV file
            required_columns: Required column types for processing
            
        Returns:
            Processed dataframe with standardized column names
            
        Raises:
            ValueError: When required columns cannot be mapped or file cannot be processed
        """
        if required_columns is None:
            required_columns = [
                GSCColumnType.QUERY, 
                GSCColumnType.URL, 
                GSCColumnType.CLICKS
            ]
        
        logger.info(f"Processing GSC data from: {file_path}")
        
        try:
            # Step 1: Detect delimiter
            delimiter = self.dialect_detector.detect_delimiter(file_path)
            logger.info(f"Using delimiter: '{delimiter}'")
            
            # Step 2: Detect encoding
            encoding = self._detect_encoding(file_path)
            logger.info(f"Using encoding: {encoding}")
            
            # Step 3: Load CSV with detected parameters
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding,
                engine='python',  # More flexible parsing
                on_bad_lines='warn'  # Handle malformed lines gracefully
            )
            
            logger.info(f"Loaded CSV with shape: {df.shape}")
            logger.info(f"Original columns: {list(df.columns)}")
            
            # Step 4: Create column mapping
            column_mapping = self._create_column_mapping(df.columns, required_columns)
            
            # Step 5: Apply column mapping and validate
            df_processed = self._apply_mapping_and_validate(df, column_mapping, required_columns)
            
            logger.info(f"Successfully processed {len(df_processed):,} rows")
            return df_processed
            
        except Exception as e:
            error_msg = f"Failed to process GSC data: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _detect_encoding(self, file_path: str, sample_size: int = 8192) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(sample_size)
            
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            logger.info(f"Encoding detection: {encoding} (confidence: {confidence:.2f})")
            
            # Fallback to common encodings if confidence is low
            if confidence < 0.7:
                for fallback_encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=fallback_encoding) as f:
                            f.read(1024)  # Test read
                        logger.info(f"Using fallback encoding: {fallback_encoding}")
                        return fallback_encoding
                    except UnicodeDecodeError:
                        continue
            
            return encoding
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def _create_column_mapping(
        self, 
        csv_columns: List[str], 
        required_columns: List[GSCColumnType]
    ) -> Dict[str, str]:
        """Create flexible column mapping from CSV columns to standard names."""
        column_mapping = {}
        mapped_types = set()
        
        # Normalize column names for comparison
        normalized_columns = {
            col.lower().strip().replace('_', ' '): col 
            for col in csv_columns
        }
        
        logger.debug(f"Normalized columns: {list(normalized_columns.keys())}")
        
        # Try to match each column type
        for col_type in GSCColumnType:
            variations = self.config.COLUMN_VARIATIONS[col_type]
            matched_column = None
            
            for variation in variations:
                normalized_variation = variation.lower().strip()
                
                # Direct match
                if normalized_variation in normalized_columns:
                    matched_column = normalized_columns[normalized_variation]
                    break
                
                # Partial match (contains or is contained)
                for norm_csv_col, original_csv_col in normalized_columns.items():
                    if (normalized_variation in norm_csv_col or 
                        norm_csv_col in normalized_variation):
                        matched_column = original_csv_col
                        break
                
                if matched_column:
                    break
            
            if matched_column:
                column_mapping[matched_column] = col_type.value
                mapped_types.add(col_type)
                logger.debug(f"Mapped: {matched_column} -> {col_type.value}")
        
        # Check for missing required columns
        missing_required = set(required_columns) - mapped_types
        if missing_required:
            missing_names = [col.value for col in missing_required]
            available_variations = []
            
            for missing_col in missing_required:
                variations = self.config.COLUMN_VARIATIONS[missing_col]
                available_variations.append(f"{missing_col.value}: {', '.join(variations)}")
            
            error_msg = (
                f"Missing required columns: {missing_names}. "
                f"Available columns: {list(csv_columns)}. "
                f"\nSupported column name variations:\n" + 
                '\n'.join(available_variations)
            )
            raise ValueError(error_msg)
        
        return column_mapping
    
    def _apply_mapping_and_validate(
        self, 
        df: pd.DataFrame, 
        column_mapping: Dict[str, str],
        required_columns: List[GSCColumnType]
    ) -> pd.DataFrame:
        """Apply column mapping and validate data."""
        # Select and rename mapped columns
        df_mapped = df[list(column_mapping.keys())].rename(columns=column_mapping)
        
        # Data validation and cleaning
        required_col_names = [col.value for col in required_columns]
        
        # Remove rows with null values in required columns
        initial_rows = len(df_mapped)
        df_clean = df_mapped.dropna(subset=required_col_names)
        
        if len(df_clean) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(df_clean)} rows with missing required data")
        
        # Convert numeric columns
        numeric_columns = ['clicks', 'impressions', 'ctr', 'position']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows with 0 or negative clicks if clicks column exists
        if 'clicks' in df_clean.columns:
            initial_count = len(df_clean)
            df_clean = df_clean[df_clean['clicks'] > 0]
            removed_count = initial_count - len(df_clean)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} rows with 0 or negative clicks")
        
        # Clean text columns
        text_columns = ['url', 'query']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # Remove empty strings
                df_clean = df_clean[df_clean[col] != '']
        
        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after cleaning and validation")
        
        logger.info(f"Data validation complete. Final dataset: {len(df_clean):,} rows")
        return df_clean


def validate_gsc_file(file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate GSC file and return processing information.
    
    Args:
        file_path: Path to the GSC CSV file
        
    Returns:
        Tuple of (is_valid, message, file_info)
    """
    try:
        processor = EnhancedGSCProcessor()
        df = processor.process_gsc_data(file_path)
        
        file_info = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample_data": df.head(3).to_dict('records') if len(df) > 0 else [],
            "data_types": df.dtypes.to_dict()
        }
        
        return True, "File successfully validated and processed", file_info
        
    except Exception as e:
        return False, str(e), {}
