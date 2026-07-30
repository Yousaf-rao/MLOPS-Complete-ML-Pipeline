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

# Input: None | Output: OS module loaded | Example: os.path.join('a', 'b')
import os
# Input: None | Output: Logging module loaded | Example: logging.getLogger('name')
import logging
# Input: None | Output: Pandas library loaded as pd | Example: pd.read_csv('data.csv')
import pandas as pd
# Input: None | Output: YAML module loaded | Example: yaml.safe_load(file)
import yaml
# Input: None | Output: TfidfVectorizer class loaded | Example: TfidfVectorizer(max_features=500)
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. LOGGER SETUP
# ============================================================

# Input: name (str) | Output: logging.Logger object | Example: setup_logger('my_app')
def setup_logger(name: str = 'feature_engineering') -> logging.Logger:
    """Setup logger with console and file handlers."""
    
    # Input: string 'name' | Output: logger instance | Example: logging.getLogger('feature_engineering')
    logger = logging.getLogger(name)
    # Input: logging.DEBUG | Output: Sets logger threshold | Example: logger.setLevel(10)
    logger.setLevel(logging.DEBUG)
    # Input: empty list | Output: Clears existing handlers | Example: logger.handlers = []
    logger.handlers = []
    
    # Input: Format string | Output: Formatter object | Example: Formatter('%(asctime)s - %(message)s')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Input: None | Output: StreamHandler for console output | Example: logging.StreamHandler()
    console_handler = logging.StreamHandler()
    # Input: logging.DEBUG | Output: Sets console threshold | Example: console_handler.setLevel(10)
    console_handler.setLevel(logging.DEBUG)
    # Input: Formatter object | Output: Assigns format to console | Example: console_handler.setFormatter(fmt)
    console_handler.setFormatter(formatter)
    # Input: console_handler | Output: Attaches handler to logger | Example: logger.addHandler(console_handler)
    logger.addHandler(console_handler)
    
    # Input: string literal 'logs' | Output: log_dir string | Example: log_dir = 'logs'
    log_dir = 'logs'
    # Input: 'logs', exist_ok=True | Output: Creates directory if missing | Example: os.makedirs('logs', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # Input: 'logs', 'feature_engineering.log' | Output: 'logs/feature_engineering.log' | Example: os.path.join('logs', 'app.log')
    log_file_path = os.path.join(log_dir, f'{name}.log')
    
    # Input: log_file_path, mode='a' | Output: FileHandler object | Example: logging.FileHandler('logs/app.log', mode='a')
    file_handler = logging.FileHandler(log_file_path, mode='a')
    # Input: logging.DEBUG | Output: Sets file threshold | Example: file_handler.setLevel(10)
    file_handler.setLevel(logging.DEBUG)
    # Input: Formatter object | Output: Assigns format to file | Example: file_handler.setFormatter(fmt)
    file_handler.setFormatter(formatter)
    # Input: file_handler | Output: Attaches handler to logger | Example: logger.addHandler(file_handler)
    logger.addHandler(file_handler)
    
    # Input: None | Output: Configured logger object | Example: return logger
    return logger


# Input: 'feature_engineering' | Output: Global logger instance | Example: logger = setup_logger('feature_engineering')
logger = setup_logger('feature_engineering')


# ============================================================
# 2. LOAD PARAMS
# ============================================================

# Input: params_path (str) | Output: Dictionary of config parameters | Example: load_params('params.yaml')
def load_params(params_path: str = 'params.yaml') -> dict:
    """Load configuration parameters from YAML file."""
    # Input: None | Output: Begins error handling block | Example: try:
    try:
        # Input: params_path, 'r' | Output: Opened file object 'f' | Example: with open('params.yaml', 'r') as f:
        with open(params_path, 'r') as f:
            # Input: File object 'f' | Output: Dictionary of YAML data | Example: params = yaml.safe_load(f)
            params = yaml.safe_load(f)
        # Input: Debug string, params_path | Output: Log message recorded | Example: logger.debug('Loaded %s', 'params.yaml')
        logger.debug('Parameters loaded from %s', params_path)
        # Input: None | Output: Returns dictionary | Example: return {'max_features': 500}
        return params
    # Input: Exception | Output: Catches generic errors into 'e' | Example: except Exception as e:
    except Exception as e:
        # Input: Error string, params_path, e | Output: Error log recorded | Example: logger.error('Failed: %s', e)
        logger.error('Failed to load params from %s: %s', params_path, e)
        # Input: None | Output: Re-throws the error to crash explicitly | Example: raise
        raise


# ============================================================
# 3. LOAD DATA
# ============================================================

# Input: file_path (str) | Output: pandas DataFrame | Example: load_data('train.csv')
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    # Input: None | Output: Begins error handling block | Example: try:
    try:
        # Input: file_path | Output: Loaded DataFrame | Example: df = pd.read_csv('train.csv')
        df = pd.read_csv(file_path)
        # Input: {'text': ''}, inplace=True | Output: DataFrame with NaNs replaced | Example: df.fillna({'text': ''}, inplace=True)
        df.fillna({'text': ''}, inplace=True)
        # Input: Debug string, file_path | Output: Log message recorded | Example: logger.debug('Loaded %s', 'train.csv')
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        # Input: None | Output: Returns cleaned DataFrame | Example: return df
        return df
    # Input: pd.errors.ParserError | Output: Catches CSV formatting errors | Example: except pd.errors.ParserError as e:
    except pd.errors.ParserError as e:
        # Input: Error string, e | Output: Error log recorded | Example: logger.error('Parse fail: %s', e)
        logger.error('Failed to parse the CSV file: %s', e)
        # Input: None | Output: Re-throws the error | Example: raise
        raise
    # Input: Exception | Output: Catches generic runtime errors | Example: except Exception as e:
    except Exception as e:
        # Input: Error string, e | Output: Error log recorded | Example: logger.error('Unexpected: %s', e)
        logger.error('Unexpected error occurred while loading the data: %s', e)
        # Input: None | Output: Re-throws the error | Example: raise
        raise


# ============================================================
# 4. APPLY TF-IDF
# ============================================================

# Input: train_data (DataFrame), test_data (DataFrame), max_features (int) | Output: Tuple of DataFrames | Example: apply_tfidf(df1, df2, 500)
def apply_tfidf(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    max_features: int
) -> tuple:
    """
    Apply TF-IDF vectorization to train and test data.
    """
    # Input: None | Output: Begins error handling block | Example: try:
    try:
        # Input: max_features (e.g. 500) | Output: TfidfVectorizer instance | Example: vectorizer = TfidfVectorizer(max_features=500)
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Input: train_data['text'] column | Output: 1D numpy array of text | Example: X_train = ['hello', 'world']
        X_train = train_data['text'].values
        # Input: train_data['target'] column | Output: 1D numpy array of labels | Example: y_train = [1, 0]
        y_train = train_data['target'].values
        # Input: test_data['text'] column | Output: 1D numpy array of text | Example: X_test = ['hi']
        X_test = test_data['text'].values
        # Input: test_data['target'] column | Output: 1D numpy array of labels | Example: y_test = [1]
        y_test = test_data['target'].values
        
        # Input: X_train array | Output: Learned vocabulary and transformed sparse matrix | Example: X_train_bow = <sparse matrix 100x500>
        X_train_bow = vectorizer.fit_transform(X_train)
        # Input: X_test array | Output: Transformed sparse matrix using existing vocabulary | Example: X_test_bow = <sparse matrix 20x500>
        X_test_bow = vectorizer.transform(X_test)
        
        # Input: X_train_bow dense array | Output: pandas DataFrame of features | Example: train_df = pd.DataFrame([[0.5, 0.0]])
        train_df = pd.DataFrame(X_train_bow.toarray())
        # Input: y_train array | Output: 'label' column appended to train_df | Example: train_df['label'] = [1, 0]
        train_df['label'] = y_train
        
        # Input: X_test_bow dense array | Output: pandas DataFrame of features | Example: test_df = pd.DataFrame([[0.5, 0.0]])
        test_df = pd.DataFrame(X_test_bow.toarray())
        # Input: y_test array | Output: 'label' column appended to test_df | Example: test_df['label'] = [1]
        test_df['label'] = y_test
        
        # Input: Debug string, shapes | Output: Log message recorded | Example: logger.debug('Shapes: %s', (100,500))
        logger.debug(
            'TF-IDF applied: train shape=%s, test shape=%s', 
            X_train_bow.shape, X_test_bow.shape
        )
        # Input: train_df, test_df | Output: Returns the two formatted DataFrames | Example: return train_df, test_df
        return train_df, test_df
        
    # Input: Exception | Output: Catches generic runtime errors | Example: except Exception as e:
    except Exception as e:
        # Input: Error string, e | Output: Error log recorded | Example: logger.error('TF-IDF failed: %s', e)
        logger.error('Error during TF-IDF transformation: %s', e)
        # Input: None | Output: Re-throws the error | Example: raise
        raise


# ============================================================
# 5. SAVE DATA
# ============================================================

# Input: df (DataFrame), file_path (str) | Output: None | Example: save_data(df, 'data.csv')
def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to a CSV file.
    """
    # Input: None | Output: Begins error handling block | Example: try:
    try:
        # Input: directory path, exist_ok=True | Output: Creates folder if missing | Example: os.makedirs('data/processed', exist_ok=True)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Input: file_path, index=False | Output: CSV file saved to disk | Example: df.to_csv('data.csv', index=False)
        df.to_csv(file_path, index=False)
        # Input: Debug string, file_path | Output: Log message recorded | Example: logger.debug('Saved to %s', 'data.csv')
        logger.debug('Data saved to %s', file_path)
        
    # Input: Exception | Output: Catches saving errors | Example: except Exception as e:
    except Exception as e:
        # Input: Error string, e | Output: Error log recorded | Example: logger.error('Save failed: %s', e)
        logger.error('Unexpected error occurred while saving the data: %s', e)
        # Input: None | Output: Re-throws the error | Example: raise
        raise


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

# Input: None | Output: None | Example: main()
def main():
    """Main feature engineering pipeline."""
    # Input: None | Output: Begins error handling block | Example: try:
    try:
        # Input: 'params.yaml' | Output: Dictionary of parameters | Example: params = {'feature_engineering': {'max_features': 500}}
        params = load_params(params_path='params.yaml')
        # Input: params dictionary | Output: integer max_features | Example: max_features = 500
        max_features = params['feature_engineering']['max_features']
        
        # Input: './data/interim/train_processed.csv' | Output: train_data DataFrame | Example: train_data = load_data('train.csv')
        train_data = load_data('./data/interim/train_processed.csv')
        # Input: './data/interim/test_processed.csv' | Output: test_data DataFrame | Example: test_data = load_data('test.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        # Input: train_data, test_data, max_features | Output: Tuple of TF-IDF DataFrames | Example: train_df, test_df = apply_tfidf(df1, df2, 500)
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        
        # Input: train_df, save path | Output: Saves train_df to CSV | Example: save_data(train_df, 'train_tfidf.csv')
        save_data(
            train_df, 
            os.path.join('./data', 'processed', 'train_tfidf.csv')
        )
        # Input: test_df, save path | Output: Saves test_df to CSV | Example: save_data(test_df, 'test_tfidf.csv')
        save_data(
            test_df, 
            os.path.join('./data', 'processed', 'test_tfidf.csv')
        )
        
        # Input: Success string | Output: Info log recorded | Example: logger.info('Done!')
        logger.info('Feature engineering completed successfully!')
        
    # Input: Exception | Output: Catches generic runtime errors | Example: except Exception as e:
    except Exception as e:
        # Input: Error string, e | Output: Error log recorded | Example: logger.error('Pipeline failed: %s', e)
        logger.error('Failed to complete the feature engineering process: %s', e)
        # Input: Error string | Output: Prints error to standard output | Example: print("Error: ...")
        print(f"Error: {e}")

# Input: __name__ variable | Output: Evaluates to True if run as script | Example: if '__main__' == '__main__':
if __name__ == '__main__':
    # Input: None | Output: Executes the main pipeline | Example: main()
    main()
