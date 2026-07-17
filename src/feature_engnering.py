"""
Feature Engineering Pipeline
============================
Complete code for:
1. Logging setup (console + file)
2. Loading CSV data
3. Applying TF-IDF transformation
4. Saving processed data
5. Main orchestration

Usage:
    python feature_engineering.py

Requires:
    - pandas
    - scikit-learn
    - PyYAML
    - params.yaml (config file)
"""

# Input: None. Output: OS module loaded for file path operations.
import os
# Input: None. Output: Logging module loaded for tracking events.
import logging
# Input: None. Output: Pandas library loaded as pd for data manipulation.
import pandas as pd
# Input: None. Output: YAML module loaded for parsing config files.
import yaml
# Input: None. Output: TfidfVectorizer class loaded for text vectorization.
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. LOGGER SETUP
# ============================================================

# Input: name (str, default 'feature_engineering'). Output: configured logging.Logger object.
def setup_logger(name: str = 'feature_engineering') -> logging.Logger:
    """Setup logger with console and file handlers."""
    
    # Input: name string. Output: logger instance for the given name.
    logger = logging.getLogger(name)
    # Input: logger level constant. Output: logger minimum level set to DEBUG.
    logger.setLevel(logging.DEBUG)
    # Input: empty list. Output: clears any existing handlers to prevent duplicate logs.
    logger.handlers = []
    
    # Input: format string. Output: Formatter object defining the log output structure.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Input: None. Output: StreamHandler object for outputting logs to the console.
    console_handler = logging.StreamHandler()
    # Input: logger level constant. Output: console handler minimum level set to DEBUG.
    console_handler.setLevel(logging.DEBUG)
    # Input: Formatter object. Output: console handler format set.
    console_handler.setFormatter(formatter)
    # Input: console_handler. Output: logger instance now includes console handler.
    logger.addHandler(console_handler)
    
    # Input: string literal 'logs'. Output: log_dir string variable.
    log_dir = 'logs'
    # Input: log_dir path and exist_ok=True. Output: creates 'logs' directory if it does not exist.
    os.makedirs(log_dir, exist_ok=True)
    # Input: log_dir and filename string. Output: full path string to the log file.
    log_file_path = os.path.join(log_dir, f'{name}.log')
    
    # Input: log_file_path string and mode='a'. Output: FileHandler object for appending logs to file.
    file_handler = logging.FileHandler(log_file_path, mode='a')
    # Input: logger level constant. Output: file handler minimum level set to DEBUG.
    file_handler.setLevel(logging.DEBUG)
    # Input: Formatter object. Output: file handler format set.
    file_handler.setFormatter(formatter)
    # Input: file_handler. Output: logger instance now includes file handler.
    logger.addHandler(file_handler)
    
    # Input: None. Output: fully configured logger object returned.
    return logger


# Input: string 'feature_engineering'. Output: initialized global logger object.
logger = setup_logger('feature_engineering')


# ============================================================
# 2. LOAD PARAMS
# ============================================================

# Input: params_path (str, default 'params.yaml'). Output: dictionary of parameters.
def load_params(params_path: str = 'params.yaml') -> dict:
    """Load configuration parameters from YAML file."""
    # Input: None. Output: starts a try block for error handling.
    try:
        # Input: params_path and 'r' (read) mode. Output: file object 'f' opened for reading.
        with open(params_path, 'r') as f:
            # Input: file object 'f'. Output: parsed dictionary of YAML contents stored in 'params'.
            params = yaml.safe_load(f)
        # Input: debug message string and params_path. Output: debug log message emitted.
        logger.debug('Parameters loaded from %s', params_path)
        # Input: None. Output: returns the 'params' dictionary.
        return params
    # Input: Exception class. Output: catches any exception into variable 'e'.
    except Exception as e:
        # Input: error message string, params_path, and exception 'e'. Output: error log message emitted.
        logger.error('Failed to load params from %s: %s', params_path, e)
        # Input: None. Output: re-raises the caught exception.
        raise


# ============================================================
# 3. LOAD DATA
# ============================================================

# Input: file_path (str). Output: pandas DataFrame containing the loaded data.
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    # Input: None. Output: starts a try block for error handling.
    try:
        # Input: file_path string. Output: pandas DataFrame 'df' loaded from CSV.
        df = pd.read_csv(file_path)
        # Input: empty string dict for text column and inplace=True. Output: replaces all NaN values in 'text' with empty strings.
        df.fillna({'text': ''}, inplace=True)
        # Input: debug message string and file_path. Output: debug log message emitted.
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        # Input: None. Output: returns the processed DataFrame 'df'.
        return df
    # Input: ParserError class. Output: catches pandas CSV parsing errors into variable 'e'.
    except pd.errors.ParserError as e:
        # Input: error message string and exception 'e'. Output: error log message emitted.
        logger.error('Failed to parse the CSV file: %s', e)
        # Input: None. Output: re-raises the caught exception.
        raise
    # Input: Exception class. Output: catches any other exceptions into variable 'e'.
    except Exception as e:
        # Input: error message string and exception 'e'. Output: error log message emitted.
        logger.error('Unexpected error occurred while loading the data: %s', e)
        # Input: None. Output: re-raises the caught exception.
        raise


# ============================================================
# 4. APPLY TF-IDF
# ============================================================

# Input: train_data (DataFrame), test_data (DataFrame), max_features (int). Output: tuple of two DataFrames.
def apply_tfidf(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    max_features: int
) -> tuple:
    """
    Apply TF-IDF vectorization to train and test data.
    """
    # Input: None. Output: starts a try block for error handling.
    try:
        # Input: max_features integer. Output: TfidfVectorizer object initialized with max_features.
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Input: train_data['text'] column. Output: numpy array of training text stored in 'X_train'.
        X_train = train_data['text'].values
        # Input: train_data['target'] column. Output: numpy array of training labels stored in 'y_train'.
        y_train = train_data['target'].values
        # Input: test_data['text'] column. Output: numpy array of testing text stored in 'X_test'.
        X_test = test_data['text'].values
        # Input: test_data['target'] column. Output: numpy array of testing labels stored in 'y_test'.
        y_test = test_data['target'].values
        
        # Input: X_train text array. Output: sparse matrix of TF-IDF features stored in 'X_train_bow'.
        X_train_bow = vectorizer.fit_transform(X_train)
        # Input: X_test text array. Output: sparse matrix of TF-IDF features stored in 'X_test_bow'.
        X_test_bow = vectorizer.transform(X_test)
        
        # Input: dense array from X_train_bow. Output: pandas DataFrame 'train_df' with TF-IDF features.
        train_df = pd.DataFrame(X_train_bow.toarray())
        # Input: y_train labels array. Output: adds 'label' column to 'train_df'.
        train_df['label'] = y_train
        
        # Input: dense array from X_test_bow. Output: pandas DataFrame 'test_df' with TF-IDF features.
        test_df = pd.DataFrame(X_test_bow.toarray())
        # Input: y_test labels array. Output: adds 'label' column to 'test_df'.
        test_df['label'] = y_test
        
        # Input: debug message string and shape tuples. Output: debug log message emitted.
        logger.debug(
            'TF-IDF applied: train shape=%s, test shape=%s', 
            X_train_bow.shape, X_test_bow.shape
        )
        # Input: None. Output: returns tuple containing (train_df, test_df).
        return train_df, test_df
        
    # Input: Exception class. Output: catches any exception during TF-IDF into variable 'e'.
    except Exception as e:
        # Input: error message string and exception 'e'. Output: error log message emitted.
        logger.error('Error during TF-IDF transformation: %s', e)
        # Input: None. Output: re-raises the caught exception.
        raise


# ============================================================
# 5. SAVE DATA
# ============================================================

# Input: df (DataFrame) and file_path (str). Output: None.
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to a CSV file.
    """
    # Input: None. Output: starts a try block for error handling.
    try:
        # Input: directory portion of file_path and exist_ok=True. Output: creates target directory if missing.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Input: file_path and index=False. Output: writes 'df' to a CSV file without index.
        df.to_csv(file_path, index=False)
        # Input: debug message string and file_path. Output: debug log message emitted.
        logger.debug('Data saved to %s', file_path)
        
    # Input: Exception class. Output: catches any saving errors into variable 'e'.
    except Exception as e:
        # Input: error message string and exception 'e'. Output: error log message emitted.
        logger.error('Unexpected error occurred while saving the data: %s', e)
        # Input: None. Output: re-raises the caught exception.
        raise


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

# Input: None. Output: None. Executes the entire pipeline.
def main():
    """Main feature engineering pipeline."""
    # Input: None. Output: starts a try block for error handling.
    try:
        # Input: 'params.yaml'. Output: dictionary of loaded parameters stored in 'params'.
        params = load_params(params_path='params.yaml')
        # Input: 'params' dictionary. Output: integer max_features value stored.
        max_features = params['feature_engineering']['max_features']
        
        # Input: train CSV path. Output: loaded pandas DataFrame 'train_data'.
        train_data = load_data('./data/interim/train_processed.csv')
        # Input: test CSV path. Output: loaded pandas DataFrame 'test_data'.
        test_data = load_data('./data/interim/test_processed.csv')
        
        # Input: train_data, test_data, max_features. Output: transformed DataFrames (train_df, test_df).
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        
        # Input: train_df and train_tfidf.csv path. Output: saves train_df to CSV.
        save_data(
            train_df, 
            os.path.join('./data', 'processed', 'train_tfidf.csv')
        )
        # Input: test_df and test_tfidf.csv path. Output: saves test_df to CSV.
        save_data(
            test_df, 
            os.path.join('./data', 'processed', 'test_tfidf.csv')
        )
        
        # Input: success message string. Output: info log message emitted.
        logger.info('Feature engineering completed successfully!')
        
    # Input: Exception class. Output: catches any pipeline error into variable 'e'.
    except Exception as e:
        # Input: error message string and exception 'e'. Output: error log message emitted.
        logger.error('Failed to complete the feature engineering process: %s', e)
        # Input: error message string. Output: prints error to standard output.
        print(f"Error: {e}")

# Input: built-in __name__ variable. Output: evaluates if script is executed directly.
if __name__ == '__main__':
    # Input: None. Output: calls the main pipeline function.
    main()
