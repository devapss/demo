import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers, callbacks
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    DataLoader class for reading multiple files from a share drive.
    Supports filtering by file prefix and creation date range.
    Includes intelligent header detection to automatically skip extra rows.
    """
    
    def __init__(self, share_drive_path: str):
        """
        Initialize DataLoader with share drive path.
        
        Parameters:
        -----------
        share_drive_path : str
            Full path to the mounted share drive (e.g., 'Z:\\financial_data')
        """
        self.share_drive_path = share_drive_path
        if not os.path.exists(share_drive_path):
            raise ValueError(f"Share drive path does not exist: {share_drive_path}")
    
    def _detect_header_row(self, file_path: str, file_extension: str = 'csv', max_rows_to_check: int = 20) -> int:
        """
        Intelligently detect where the actual data starts in a file.
        
        This method analyzes the file structure to find the row where actual tabular data begins.
        It looks for patterns that indicate a header row followed by data rows.
        
        Parameters:
        -----------
        file_path : str
            Full path to the file
        file_extension : str
            File extension ('csv' or 'xlsx')
        max_rows_to_check : int
            Maximum number of rows to analyze (default: 20)
        
        Returns:
        --------
        int
            Number of rows to skip before the actual header
        """
        try:
            # Read first N rows without parsing
            if file_extension == 'csv':
                # Read as raw text lines
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [f.readline() for _ in range(max_rows_to_check)]
            elif file_extension == 'xlsx':
                # Read Excel without headers
                temp_df = pd.read_excel(file_path, header=None, nrows=max_rows_to_check)
                lines = temp_df.values.tolist()
            else:
                return 0
            
            best_header_row = 0
            max_score = -1
            
            for idx, line in enumerate(lines):
                if not line or (isinstance(line, str) and line.strip() == ''):
                    continue
                
                # Convert line to list of values
                if isinstance(line, str):
                    if file_extension == 'csv':
                        # Try to detect delimiter
                        delimiter = ',' if ',' in line else '\t' if '\t' in line else ';'
                        values = [v.strip() for v in line.split(delimiter)]
                    else:
                        values = [line]
                else:
                    values = [str(v).strip() if v is not None else '' for v in line]
                
                # Filter out empty values
                values = [v for v in values if v and v.lower() not in ['nan', 'none', '']]
                
                if len(values) < 2:  # Need at least 2 columns
                    continue
                
                # Calculate score for this row being a header
                score = 0
                
                # 1. Check for common header keywords
                header_keywords = ['id', 'name', 'date', 'amount', 'value', 'type', 'category', 
                                  'description', 'status', 'code', 'number', 'total', 'balance',
                                  'transaction', 'account', 'customer', 'product', 'price', 'quantity']
                for val in values:
                    val_lower = str(val).lower()
                    if any(keyword in val_lower for keyword in header_keywords):
                        score += 3
                
                # 2. Check if values are mostly text (headers are usually text)
                text_count = sum(1 for v in values if not str(v).replace('.', '').replace('-', '').replace('/', '').isdigit())
                if text_count / len(values) > 0.7:
                    score += 2
                
                # 3. Check for reasonable column count (3-100 columns)
                if 3 <= len(values) <= 100:
                    score += 2
                
                # 4. Penalize if row has too many numbers (likely data, not header)
                numeric_count = sum(1 for v in values if str(v).replace('.', '').replace('-', '').isdigit())
                if numeric_count / len(values) > 0.5:
                    score -= 3
                
                # 5. Check if next row exists and looks like data
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                    if next_line:
                        if isinstance(next_line, str):
                            delimiter = ',' if ',' in next_line else '\t' if '\t' in next_line else ';'
                            next_values = [v.strip() for v in next_line.split(delimiter)]
                        else:
                            next_values = [str(v).strip() if v is not None else '' for v in next_line]
                        
                        next_values = [v for v in next_values if v and v.lower() not in ['nan', 'none', '']]
                        
                        # Check if next row has similar column count
                        if len(next_values) > 0 and abs(len(values) - len(next_values)) <= 2:
                            score += 2
                            
                            # Check if next row has more numbers (data characteristic)
                            next_numeric_count = sum(1 for v in next_values if str(v).replace('.', '').replace('-', '').replace('/', '').isdigit())
                            if next_numeric_count / len(next_values) > 0.3:
                                score += 3
                
                # 6. Penalize very early rows (often metadata)
                if idx == 0 and len(values) < 3:
                    score -= 2
                
                # 7. Check for delimiters consistency
                if isinstance(line, str):
                    delimiter_count = line.count(',') + line.count('\t') + line.count(';')
                    if delimiter_count >= 2:
                        score += 1
                
                # Update best header row
                if score > max_score:
                    max_score = score
                    best_header_row = idx
            
            print(f"  Detected header at row {best_header_row} (score: {max_score})")
            return best_header_row
            
        except Exception as e:
            print(f"  Warning: Header detection failed ({str(e)}), using row 0")
            return 0
    
    def read_files_by_prefix_and_date(self, 
                                      file_prefix: str,
                                      start_date: str,
                                      end_date: str,
                                      skip_rows: Optional[int] = None,
                                      file_extension: str = 'csv',
                                      auto_detect_header: bool = True) -> pd.DataFrame:
        """
        Read multiple files from share drive matching prefix and date range.
        
        Parameters:
        -----------
        file_prefix : str
            Prefix to filter files (e.g., 'financial_report_')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        skip_rows : int, optional
            Number of rows to skip at the beginning of each file. 
            If None and auto_detect_header=True, will auto-detect.
        file_extension : str
            File extension to look for ('csv' or 'xlsx', default: 'csv')
        auto_detect_header : bool
            If True, automatically detect where data starts (default: True)
        
        Returns:
        --------
        pd.DataFrame
            Merged dataframe from all matching files
        """
        print(f"Scanning share drive: {self.share_drive_path}")
        print(f"Looking for files with prefix: {file_prefix}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Auto-detect header: {auto_detect_header}")
        print("=" * 60)
        
        # Convert date strings to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Find all matching files
        matching_files = []
        for root, dirs, files in os.walk(self.share_drive_path):
            for file in files:
                if file.startswith(file_prefix) and file.endswith(f'.{file_extension}'):
                    file_path = os.path.join(root, file)
                    
                    # Get file creation time
                    creation_time = os.path.getctime(file_path)
                    file_date = datetime.fromtimestamp(creation_time)
                    
                    # Check if file is within date range
                    if start_dt <= file_date <= end_dt:
                        matching_files.append({
                            'path': file_path,
                            'name': file,
                            'created': file_date
                        })
        
        if not matching_files:
            raise ValueError(f"No files found matching prefix '{file_prefix}' in date range {start_date} to {end_date}")
        
        # Sort files by creation date
        matching_files.sort(key=lambda x: x['created'])
        
        print(f"Found {len(matching_files)} matching files:")
        for f in matching_files:
            print(f"  - {f['name']} (Created: {f['created'].strftime('%Y-%m-%d %H:%M:%S')})")
        print("=" * 60)
        
        # Read and merge all files
        dataframes = []
        for file_info in matching_files:
            print(f"Reading: {file_info['name']}...")
            try:
                # Determine rows to skip
                if auto_detect_header and skip_rows is None:
                    rows_to_skip = self._detect_header_row(file_info['path'], file_extension)
                else:
                    rows_to_skip = skip_rows if skip_rows is not None else 0
                
                # Read file
                if file_extension == 'csv':
                    df = pd.read_csv(file_info['path'], skiprows=rows_to_skip)
                elif file_extension == 'xlsx':
                    df = pd.read_excel(file_info['path'], skiprows=rows_to_skip)
                else:
                    raise ValueError(f"Unsupported file extension: {file_extension}")
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Remove completely empty rows
                df = df.dropna(how='all')
                
                # Remove columns that are completely empty
                df = df.dropna(axis=1, how='all')
                
                # Add source file column for tracking
                df['_source_file'] = file_info['name']
                df['_file_created_date'] = file_info['created']
                
                dataframes.append(df)
                print(f"  Loaded {len(df)} rows × {len(df.columns)} columns (skipped {rows_to_skip} rows)")
            except Exception as e:
                print(f"  ERROR reading {file_info['name']}: {str(e)}")
                continue
        
        if not dataframes:
            raise ValueError("Failed to read any files successfully")
        
        # Merge all dataframes
        print("=" * 60)
        print("Merging all dataframes...")
        
        # Check if all dataframes have same columns
        first_cols = set(dataframes[0].columns)
        all_same_cols = all(set(df.columns) == first_cols for df in dataframes)
        
        if not all_same_cols:
            print("Warning: Files have different columns. Merging with outer join.")
            merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
        else:
            merged_df = pd.concat(dataframes, ignore_index=True)
        
        print(f"Total rows after merging: {len(merged_df)}")
        print(f"Total columns: {len(merged_df.columns)}")
        print(f"Column names: {list(merged_df.columns)[:10]}{'...' if len(merged_df.columns) > 10 else ''}")
        print("=" * 60)
        
        return merged_df


class DataPreprocessor:
    """
    DataPreprocessor class for handling numerical and categorical columns.
    Prepares data for autoencoder training by encoding categorical variables
    and scaling numerical features.
    """
    
    def __init__(self):
        """Initialize DataPreprocessor with empty encoders dictionary."""
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.all_columns = []
    
    def identify_column_types(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> Dict[str, List[str]]:
        """
        Automatically identify numerical and categorical columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        exclude_cols : List[str]
            List of column names to exclude from processing (e.g., metadata columns)
        
        Returns:
        --------
        Dict with 'numerical' and 'categorical' column lists
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Filter out excluded columns
        cols_to_process = [col for col in df.columns if col not in exclude_cols]
        
        numerical = []
        categorical = []
        
        for col in cols_to_process:
            if df[col].dtype in ['int64', 'float64']:
                numerical.append(col)
            elif df[col].dtype in ['object', 'category', 'bool']:
                categorical.append(col)
        
        self.numerical_columns = numerical
        self.categorical_columns = categorical
        self.all_columns = numerical + categorical
        
        print("Column Type Identification:")
        print(f"  Numerical columns ({len(numerical)}): {numerical[:5]}{'...' if len(numerical) > 5 else ''}")
        print(f"  Categorical columns ({len(categorical)}): {categorical[:5]}{'...' if len(categorical) > 5 else ''}")
        print(f"  Excluded columns: {exclude_cols}")
        print("=" * 60)
        
        return {'numerical': numerical, 'categorical': categorical}
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            If True, fit the encoders. If False, use existing encoders (for new data)
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with encoded categorical features
        """
        df_encoded = df.copy()
        
        if not self.categorical_columns:
            return df_encoded
        
        print("Encoding categorical features...")
        
        for col in self.categorical_columns:
            if col not in df_encoded.columns:
                continue
            
            # Handle missing values
            df_encoded[col] = df_encoded[col].fillna('MISSING')
            
            if fit:
                # Fit new encoder
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} unique categories")
            else:
                # Use existing encoder
                if col not in self.label_encoders:
                    raise ValueError(f"No encoder found for column '{col}'. Run with fit=True first.")
                
                le = self.label_encoders[col]
                
                # Handle unseen categories
                def safe_transform(val):
                    try:
                        return le.transform([str(val)])[0]
                    except ValueError:
                        # Return the most common category or -1 for unseen values
                        return -1
                
                df_encoded[col] = df_encoded[col].astype(str).apply(safe_transform)
        
        print("=" * 60)
        return df_encoded
    
    def preprocess_for_autoencoder(self, 
                                   df: pd.DataFrame,
                                   exclude_cols: List[str] = None,
                                   fit: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for autoencoder.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        exclude_cols : List[str]
            Columns to exclude from processing
        fit : bool
            Whether to fit encoders (True for training data, False for new data)
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed dataframe ready for autoencoder
        """
        if exclude_cols is None:
            exclude_cols = ['_source_file', '_file_created_date']
        
        print("Starting preprocessing pipeline...")
        print("=" * 60)
        
        # Identify column types
        if fit:
            self.identify_column_types(df, exclude_cols)
        
        # Select only the columns we identified during fit
        df_processed = df[self.all_columns].copy()
        
        # Handle missing values in numerical columns
        for col in self.numerical_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed, fit=fit)
        
        # Convert all columns to numeric (in case any encoding produced non-numeric)
        df_processed = df_processed.apply(pd.to_numeric, errors='coerce')
        
        # Fill any remaining NaN values
        df_processed = df_processed.fillna(0)
        
        print(f"Preprocessing complete. Final shape: {df_processed.shape}")
        print("=" * 60)
        
        return df_processed


class FileComparator:
    """
    FileComparator class for comparing two CSV or Excel files.
    Generates detailed mismatch report in text format.
    """
    
    def __init__(self):
        """Initialize FileComparator."""
        pass
    
    def read_file(self, file_path: str) -> pd.DataFrame:
        """
        Read CSV or Excel file.
        
        Parameters:
        -----------
        file_path : str
            Full path to the file
        
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .csv, .xlsx, or .xls")
    
    def compare_files(self, 
                      file1_path: str, 
                      file2_path: str, 
                      output_path: str = 'comparison_report.txt',
                      tolerance: float = 1e-6) -> bool:
        """
        Compare two CSV or Excel files and generate detailed mismatch report.
        
        Parameters:
        -----------
        file1_path : str
            Full path to first file
        file2_path : str
            Full path to second file
        output_path : str
            Path for output mismatch report (default: 'comparison_report.txt')
        tolerance : float
            Numerical tolerance for floating point comparisons (default: 1e-6)
        
        Returns:
        --------
        bool
            True if files are identical, False otherwise
        """
        print("=" * 70)
        print("FILE COMPARISON REPORT")
        print("=" * 70)
        print(f"File 1: {file1_path}")
        print(f"File 2: {file2_path}")
        print("=" * 70)
        
        # Read both files
        try:
            df1 = self.read_file(file1_path)
            df2 = self.read_file(file2_path)
        except Exception as e:
            error_msg = f"Error reading files: {str(e)}"
            print(error_msg)
            with open(output_path, 'w') as f:
                f.write(error_msg)
            return False
        
        mismatches = []
        files_identical = True
        
        # Check 1: Shape comparison
        if df1.shape != df2.shape:
            files_identical = False
            mismatch = f"SHAPE MISMATCH:\n"
            mismatch += f"  File 1: {df1.shape[0]} rows × {df1.shape[1]} columns\n"
            mismatch += f"  File 2: {df2.shape[0]} rows × {df2.shape[1]} columns\n"
            mismatches.append(mismatch)
            print(mismatch)
        
        # Check 2: Column names comparison
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        
        if cols1 != cols2:
            files_identical = False
            mismatch = "COLUMN MISMATCH:\n"
            
            only_in_file1 = cols1 - cols2
            only_in_file2 = cols2 - cols1
            
            if only_in_file1:
                mismatch += f"  Columns only in File 1: {list(only_in_file1)}\n"
            if only_in_file2:
                mismatch += f"  Columns only in File 2: {list(only_in_file2)}\n"
            
            mismatches.append(mismatch)
            print(mismatch)
        
        # Check 3: Data comparison (only for common columns)
        common_columns = list(cols1.intersection(cols2))
        
        if common_columns and df1.shape[0] == df2.shape[0]:
            print("Comparing data values...")
            
            for col in common_columns:
                col_mismatches = []
                
                # Check dtype compatibility
                if df1[col].dtype != df2[col].dtype:
                    files_identical = False
                    col_mismatches.append(f"  Data type mismatch: {df1[col].dtype} vs {df2[col].dtype}")
                
                # Compare values
                for idx in range(len(df1)):
                    val1 = df1.loc[idx, col]
                    val2 = df2.loc[idx, col]
                    
                    # Handle NaN comparisons
                    if pd.isna(val1) and pd.isna(val2):
                        continue
                    elif pd.isna(val1) or pd.isna(val2):
                        files_identical = False
                        col_mismatches.append(
                            f"  Row {idx}: '{val1}' (File1) vs '{val2}' (File2)"
                        )
                        continue
                    
                    # Numerical comparison with tolerance
                    if df1[col].dtype in ['int64', 'float64'] and df2[col].dtype in ['int64', 'float64']:
                        if abs(float(val1) - float(val2)) > tolerance:
                            files_identical = False
                            col_mismatches.append(
                                f"  Row {idx}: {val1} (File1) vs {val2} (File2), diff={abs(float(val1) - float(val2)):.2e}"
                            )
                    # String/Object comparison
                    else:
                        if str(val1) != str(val2):
                            files_identical = False
                            col_mismatches.append(
                                f"  Row {idx}: '{val1}' (File1) vs '{val2}' (File2)"
                            )
                
                if col_mismatches:
                    mismatch = f"\nCOLUMN '{col}' MISMATCHES ({len(col_mismatches)} differences):\n"
                    # Limit to first 100 mismatches per column
                    if len(col_mismatches) > 100:
                        mismatch += '\n'.join(col_mismatches[:100])
                        mismatch += f"\n  ... and {len(col_mismatches) - 100} more mismatches\n"
                    else:
                        mismatch += '\n'.join(col_mismatches) + '\n'
                    
                    mismatches.append(mismatch)
                    print(f"Found {len(col_mismatches)} mismatches in column '{col}'")
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("FILE COMPARISON REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"File 1: {file1_path}\n")
            f.write(f"File 2: {file2_path}\n")
            f.write(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            if files_identical:
                f.write("✓ FILES ARE IDENTICAL\n\n")
                f.write("Both files have:\n")
                f.write(f"  - Same shape: {df1.shape[0]} rows × {df1.shape[1]} columns\n")
                f.write(f"  - Same columns: {list(df1.columns)}\n")
                f.write(f"  - Identical data values\n")
            else:
                f.write("✗ FILES ARE DIFFERENT\n\n")
                f.write(f"Total mismatches found: {len(mismatches)}\n\n")
                f.write("=" * 70 + "\n\n")
                
                for i, mismatch in enumerate(mismatches, 1):
                    f.write(f"MISMATCH #{i}:\n")
                    f.write(mismatch)
                    f.write("\n" + "=" * 70 + "\n\n")
        
        print("=" * 70)
        if files_identical:
            print("✓ FILES ARE IDENTICAL")
        else:
            print(f"✗ FILES ARE DIFFERENT - {len(mismatches)} mismatch(es) found")
        print(f"Detailed report saved to: {output_path}")
        print("=" * 70)
        
        return files_identical


class AutoEncoder:
    """
    AutoEncoder class for anomaly detection on financial data.
    Uses deep learning autoencoder to identify anomalous patterns.
    """
    
    def __init__(self, input_dim=None, hidden_layer_sizes=None, activation='relu', date_col=None):
        """
        Initialize AutoEncoder.
        
        Parameters:
        -----------
        input_dim : int, optional
            Input dimension (automatically determined during fit)
        hidden_layer_sizes : list, optional
            List of hidden layer sizes (automatically determined if not provided)
        activation : str
            Activation function for hidden layers (default: 'relu')
        date_col : str, optional
            Name of date column for temporal feature engineering
        """
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.normal_stats = None
        self.date_col = date_col
        self.anomaly_threshold = None
    
    def prepare_data(self, X):
        """
        Given a DataFrame and a date column, create cyclic features for date/time.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with temporal features appended
        """
        df = X.copy()
        
        # Create temporal features if date column specified
        if self.date_col is not None and self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            
            # Basic date features
            df['month'] = df[self.date_col].dt.month
            df['day'] = df[self.date_col].dt.day
            df['day_of_week'] = df[self.date_col].dt.dayofweek
            df['week_of_year'] = df[self.date_col].dt.isocalendar().week.astype(int)
            df['day_of_year'] = df[self.date_col].dt.dayofyear
            
            # Cyclic encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            # Optional flags
            df['is_month_end'] = df[self.date_col].dt.is_month_end.astype(int)
            df['is_month_start'] = df[self.date_col].dt.is_month_start.astype(int)
            df['is_quarter_end'] = df[self.date_col].dt.is_quarter_end.astype(int)
            df['is_quarter_start'] = df[self.date_col].dt.is_quarter_start.astype(int)
            
            df.drop(self.date_col, axis=1, inplace=True)
        
        return df
    
    def _calculate_normal_stats(self, X_normal):
        """
        Calculate statistics for normal data to help with explanations.
        
        Parameters:
        -----------
        X_normal : np.ndarray
            Scaled normal training data
        """
        stats = {}
        for i, feature in enumerate(self.feature_names):
            stats[feature] = {
                'mean': np.mean(X_normal[:, i]),
                'std': np.std(X_normal[:, i]),
                'q25': np.percentile(X_normal[:, i], 25),
                'q75': np.percentile(X_normal[:, i], 75),
                'min': np.min(X_normal[:, i]),
                'max': np.max(X_normal[:, i]),
                'median': np.median(X_normal[:, i])
            }
        self.normal_stats = stats
    
    def _setup_dimension(self, df):
        """
        Setup input dimension and hidden layer sizes based on data.
        
        Parameters:
        -----------
        df : np.ndarray
            Training data array
        """
        self.input_dim = df.shape[1]
        self.model = None
        
        if self.hidden_layer_sizes is None:
            # Determine hidden layer sizes based on input_dim
            sizes = []
            current_size = self.input_dim
            while current_size > 4:
                current_size = max(4, current_size // 2)
                sizes.append(current_size)
            self.hidden_layer_sizes = sizes
    
    def _decide_anomaly_threshold(self, errors, contamination=0.05):
        """
        Given training residual errors, compute threshold for anomalies.
        
        Parameters:
        -----------
        errors : np.ndarray
            Reconstruction errors
        contamination : float
            Expected proportion of anomalies (default: 0.05)
        """
        self.anomaly_threshold = np.percentile(errors, 100 * (1 - contamination))
        print(f"Anomaly threshold: {self.anomaly_threshold}")
        print("=" * 60)
    
    def build(self):
        """
        Build the autoencoder architecture using the hidden_layer_sizes.
        """
        input_layer = keras.Input(shape=(self.input_dim,))
        x = input_layer
        
        # Encoder
        for size in self.hidden_layer_sizes:
            x = layers.Dense(size, activation=self.activation)(x)
        
        # Decoder (mirror architecture)
        for size in reversed(self.hidden_layer_sizes[:-1]):
            x = layers.Dense(size, activation=self.activation)(x)
        
        # Output layer
        output_layer = layers.Dense(self.input_dim, activation='linear')(x)
        
        self.model = keras.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=optimizers.Adam(), loss='mse')
    
    def fit(self, X, epochs=500, batch_size=128, patience=20):
        """
        Train the autoencoder on normal data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data (should be normal/non-anomalous)
        epochs : int
            Maximum training epochs (default: 500)
        batch_size : int
            Batch size for training (default: 128)
        patience : int
            Early stopping patience (default: 20)
        """
        print("Preparing data set with temporal features...")
        df = self.prepare_data(X)
        print(df.head())
        print("=" * 60)
        
        print("Setting up feature list...")
        self.feature_names = df.columns.tolist()
        print(f"Features ({len(self.feature_names)}): {self.feature_names[:10]}{'...' if len(self.feature_names) > 10 else ''}")
        print("=" * 60)
        
        print("Scaling features...")
        df = self.scaler.fit_transform(df)
        print("=" * 60)
        
        print("Calculating normal statistics...")
        self._calculate_normal_stats(df)
        print("=" * 60)
        
        print("Setting up model dimensions...")
        self._setup_dimension(df)
        print(f"Input dimension: {self.input_dim}")
        print(f"Hidden layer sizes: {self.hidden_layer_sizes}")
        print("=" * 60)
        
        print("Building model...")
        if self.model is None:
            self.build()
        print(self.model.summary())
        print("=" * 60)
        
        X_train, X_test = train_test_split(df, test_size=0.3, shuffle=False)
        
        print("Training model...")
        es = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        history = self.model.fit(
            X_train, X_train,
            validation_data=(X_test, X_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )
        print("=" * 60)
        
        print("Setting up anomaly threshold...")
        # Compute reconstruction error on training data for threshold
        train_residuals, _ = self.predict(X)
        self._decide_anomaly_threshold(train_residuals, contamination=0)
        print("=" * 60)
    
    def predict(self, X):
        """
        Reconstruct data and compute residuals.
        
        Parameters:
        -----------
        X : pd.DataFrame
            New data to predict
        
        Returns:
        --------
        residuals : np.ndarray
            Reconstruction errors
        X_pred : np.ndarray
            Reconstructed data
        """
        print("Preparing data set with temporal features...")
        df = self.prepare_data(X)
        
        print("Arranging columns as per training data...")
        df = df[self.feature_names]
        
        print("Scaling data...")
        df = self.scaler.transform(df)
        
        print("Predicting reconstruction...")
        X_pred = self.model.predict(df, verbose=0)
        residuals = np.mean((df - X_pred) ** 2, axis=1)
        
        return residuals, X_pred
    
    def _calculate_risk_level(self, anomaly_score):
        """
        Calculate risk level based on anomaly score.
        
        Parameters:
        -----------
        anomaly_score : float
            Reconstruction error for a sample
        
        Returns:
        --------
        str
            Risk level ('LOW', 'NORMAL', 'MEDIUM', 'HIGH')
        """
        if self.anomaly_threshold == 0:
            return 'NORMAL'
        
        relative_diff = (abs(anomaly_score) - abs(self.anomaly_threshold)) / abs(self.anomaly_threshold)
        
        if relative_diff <= 0.1:
            return 'LOW'
        elif relative_diff <= 0.3:
            return 'NORMAL'
        elif relative_diff <= 0.5:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _explain_single_anomaly(self, x_original, x_scaled):
        """
        Explain why a single data point is anomalous.
        Analyzes each feature individually.
        
        Parameters:
        -----------
        x_original : pd.Series
            Original unscaled data point
        x_scaled : np.ndarray
            Scaled data point
        
        Returns:
        --------
        dict
            Detailed explanations for each feature
        """
        explanations = {}
        
        for i, feature in enumerate(self.feature_names):
            feature_value_scaled = x_scaled[i]
            feature_value_original = x_original[feature]
            normal_stats = self.normal_stats[feature]
            
            # Calculate how far the feature deviates from normal
            z_score = (feature_value_scaled - normal_stats['mean']) / (normal_stats['std'] + 1e-8)
            
            # Determine if feature is anomalous
            is_anomalous = False
            reason = ""
            
            if abs(z_score) > 2:  # More than 2 standard deviations
                is_anomalous = True
                if z_score > 2:
                    reason = f"Extremely high value ({feature_value_original:.4f}) - {abs(z_score):.2f} std above normal mean"
                else:
                    reason = f"Extremely low value ({feature_value_original:.4f}) - {abs(z_score):.2f} std below normal mean"
            elif (feature_value_scaled < normal_stats['q25'] - 1.5 * (normal_stats['q75'] - normal_stats['q25']) or
                  feature_value_scaled > normal_stats['q75'] + 1.5 * (normal_stats['q75'] - normal_stats['q25'])):
                is_anomalous = True
                reason = f"Outlier value ({feature_value_original:.4f}) - outside normal interquartile range"
            
            explanations[feature] = {
                'is_anomalous': is_anomalous,
                'value_original': feature_value_original,
                'value_scaled': feature_value_scaled,
                'z_score': z_score,
                'reason': reason,
                'normal_mean': normal_stats['mean'],
                'normal_std': normal_stats['std'],
                'confidence': min(abs(z_score) / 3.0, 1.0) if is_anomalous else 0.0
            }
        
        return explanations
    
    def analyze_new_data_cellwise(self, new_data):
        """
        Analyze new data with comprehensive cell-wise anomaly detection.
        
        Parameters:
        -----------
        new_data : pd.DataFrame
            New data to analyze
        
        Returns:
        --------
        results : list or str
            Detailed cell-wise analysis results or "No anomaly found"
        """
        anomaly_scores, _ = self.predict(new_data)
        
        # Create results DataFrame
        results_df = new_data.copy()
        results_df = self.prepare_data(results_df)
        results_df['reconstruction_error'] = anomaly_scores
        results_df['is_anomaly'] = (anomaly_scores > self.anomaly_threshold).astype(int)
        
        # Save intermediate results
        results_df.to_csv("Anomalies_result.csv", index=False)
        
        results = []
        results_df['risk_level'] = ''
        results_df['anomalous_cells'] = ''
        results_df['normal_cells'] = ''
        
        filtered_df = results_df[results_df['is_anomaly'] == 1]
        
        if len(filtered_df) < 1:
            return 'No anomaly found'
        
        print(f"Found {len(filtered_df)} anomalies. Analyzing...")
        
        for index, row in filtered_df.iterrows():
            row_result = {
                'row_index': index,
                'is_anomaly': row['is_anomaly'],
                'anomaly_score': row['reconstruction_error'],
                'anomalous_cells': [],
                'normal_cells': [],
                'risk_level': self._calculate_risk_level(row['reconstruction_error'])
            }
            
            results_df.at[index, 'risk_level'] = row_result['risk_level']
            
            # Get explanation for this row
            row_numerical = row[self.feature_names]
            row_np = row_numerical.to_numpy()
            row_reshaped = row_np.reshape(1, -1)
            scaled_row = self.scaler.transform(row_reshaped)
            
            explanation = self._explain_single_anomaly(row, scaled_row[0])
            
            for feature, details in explanation.items():
                cell_info = {
                    'feature': feature,
                    'value': float(details['value_original']),
                    'z_score': float(details['z_score']),
                    'reason': details['reason'] if details['reason'] else 'Normal variation',
                    'confidence': details['confidence']
                }
                
                if details['is_anomalous']:
                    row_result['anomalous_cells'].append(cell_info)
                else:
                    row_result['normal_cells'].append(cell_info)
            
            # Sort anomalous cells by confidence
            row_result['anomalous_cells'].sort(key=lambda x: x['confidence'], reverse=True)
            
            results_df.at[index, 'anomalous_cells'] = str(row_result['anomalous_cells'])
            results_df.at[index, 'normal_cells'] = str(row_result['normal_cells'])
            
            results.append(row_result)
        
        # Save final results
        results_df.to_csv("Final_Anomalies_result.csv", index=False)
        print("Results saved to 'Final_Anomalies_result.csv'")
        
        return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print("ENHANCED FINANCIAL ANOMALY DETECTION SYSTEM")
    print("=" * 70 + "\n")
    
    # Example 1: File Comparison
    print("\n### EXAMPLE 1: FILE COMPARISON ###\n")
    comparator = FileComparator()
    
    # Uncomment and modify paths to compare your files
    # comparator.compare_files(
    #     file1_path='path/to/file1.csv',
    #     file2_path='path/to/file2.csv',
    #     output_path='comparison_report.txt'
    # )
    
    # Example 2: Load data from share drive with AUTO HEADER DETECTION
    print("\n### EXAMPLE 2: LOAD DATA FROM SHARE DRIVE (AUTO HEADER DETECTION) ###\n")
    
    # Uncomment and modify to use your share drive
    loader = DataLoader(share_drive_path='Z:\\financial_data')
    
    # Method 1: Auto-detect header (RECOMMENDED)
    raw_data = loader.read_files_by_prefix_and_date(
        file_prefix='financial_report_',
        start_date='2024-01-01',
        end_date='2024-12-31',
        auto_detect_header=True  # Will automatically find where data starts
    )
    # 
    # # Method 2: Manual skip rows (if you know the exact number)
    raw_data = loader.read_files_by_prefix_and_date(
        file_prefix='financial_report_',
        start_date='2024-01-01',
        end_date='2024-12-31',
        skip_rows=5,
        auto_detect_header=False
    )
    
    # # Preprocess data
    # preprocessor = DataPreprocessor()
    # processed_data = preprocessor.preprocess_for_autoencoder(
    #     df=raw_data,
    #     exclude_cols=['_source_file', '_file_created_date'],
    #     fit=True
    # )
    
    # # Train autoencoder
    # autoencoder = AutoEncoder(date_col='transaction_date')
    # autoencoder.fit(processed_data, epochs=100, batch_size=64, patience=10)
    
    # # Analyze new data
    # new_raw_data = loader.read_files_by_prefix_and_date(
    #     file_prefix='financial_report_',
    #     start_date='2025-01-01',
    #     end_date='2025-01-31',
    #     auto_detect_header=True  # Auto-detect for new files too
    # )
    
    # new_processed_data = preprocessor.preprocess_for_autoencoder(
    #     df=new_raw_data,
    #     exclude_cols=['_source_file', '_file_created_date'],
    #     fit=False  # Use existing encoders
    # )
    
    # results = autoencoder.analyze_new_data_cellwise(new_processed_data)
    # print("\nAnomaly Detection Results:")
    # print(results)
    
    # Example 3: Simple demo with generated data
    print("\n### EXAMPLE 3: DEMO WITH GENERATED DATA ###\n")
    
    def generate_sample_data(n_samples=1000, n_anomalies=0):
        """Generate sample financial data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
        
        data = {
            'transaction_date': dates,
            'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
            'balance': np.random.uniform(1000, 50000, n_samples),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'volume': np.random.exponential(scale=100, size=n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Inject anomalies
        if n_anomalies > 0:
            anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)
            df.loc[anomaly_indices, 'amount'] = df['amount'].quantile(0.99) * np.random.uniform(3, 10, n_anomalies)
        
        return df
    
    # Generate training data (no anomalies)
    train_data = generate_sample_data(n_samples=2000, n_anomalies=0)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    processed_train = preprocessor.preprocess_for_autoencoder(
        df=train_data,
        exclude_cols=['transaction_date'],
        fit=True
    )
    
    # Add back date column for temporal features
    processed_train['transaction_date'] = train_data['transaction_date'].values
    
    # Train autoencoder
    autoencoder = AutoEncoder(date_col='transaction_date')
    autoencoder.fit(processed_train, epochs=50, batch_size=32, patience=5)
    
    # Generate test data with anomalies
    test_data = generate_sample_data(n_samples=500, n_anomalies=25)
    
    # Preprocess test data
    processed_test = preprocessor.preprocess_for_autoencoder(
        df=test_data,
        exclude_cols=['transaction_date'],
        fit=False
    )
    
    # Add back date column
    processed_test['transaction_date'] = test_data['transaction_date'].values
    
    # Detect anomalies
    results = autoencoder.analyze_new_data_cellwise(processed_test)
    
    if isinstance(results, str):
        print(results)
    else:
        print(f"\nDetected {len(results)} anomalies:")
        for i, result in enumerate(results[:5], 1):  # Show first 5
            print(f"\nAnomaly #{i}:")
            print(f"  Row Index: {result['row_index']}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Anomaly Score: {result['anomaly_score']:.6f}")
            print(f"  Anomalous Cells: {len(result['anomalous_cells'])}")
            if result['anomalous_cells']:
                print("  Top contributors:")
                for cell in result['anomalous_cells'][:3]:
                    print(f"    - {cell['feature']}: {cell['reason']}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
