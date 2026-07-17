import pandas as pd  # Import pandas library for data manipulation and analysis
import os  # Import os module for interacting with the operating system (e.g., file paths)
from sklearn.model_selection import train_test_split  # Import function to split data into training and testing sets
import logging  # Import logging module to track events that happen when the software runs

# Ensure the "logs" directory exists
log_dir = 'logs'  # Define the directory name where log files will be stored
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist, don't raise error if it does

# logging configuration
logger = logging.getLogger('data_ingestion')  # Create a logger named 'data_ingestion'
logger.setLevel('DEBUG')  # Set the minimum logging level to DEBUG (captures all log messages)

console_handler = logging.StreamHandler()  # Create a handler to output log messages to the console (standard output)
console_handler.setLevel('DEBUG')  # Set the console handler's logging level to DEBUG

log_file_path = os.path.join(log_dir, 'data_ingestion.log')  # Construct the full path for the log file
file_handler = logging.FileHandler(log_file_path)  # Create a handler to output log messages to a file
file_handler.setLevel('DEBUG')  # Set the file handler's logging level to DEBUG

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Define the format for log messages (time, logger name, level, message)
console_handler.setFormatter(formatter)  # Apply the format to the console handler
file_handler.setFormatter(formatter)  # Apply the format to the file handler

logger.addHandler(console_handler)  # Add the console handler to the logger
logger.addHandler(file_handler)  # Add the file handler to the logger

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)  # Read the CSV file from the provided URL into a pandas DataFrame
        logger.debug('Data loaded from %s', data_url)  # Log a debug message indicating successful loading
        return df  # Return the loaded DataFrame
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)  # Log an error message if an exception occurs
        raise  # Re-raise the exception to be handled by the caller

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)  # Rename columns 'v1' to 'target' and 'v2' to 'text' directly in the DataFrame
        logger.debug('Data preprocessing completed')  # Log a debug message indicating successful preprocessing
        return df  # Return the preprocessed DataFrame
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)  # Log an error if the expected columns are not found
        raise  # Re-raise the exception
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)  # Log any other unexpected errors
        raise  # Re-raise the exception

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')  # Construct the path for the 'raw' data directory
        os.makedirs(raw_data_path, exist_ok=True)  # Create the 'raw' directory if it doesn't exist
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)  # Save the training data to a CSV file without the index column
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)  # Save the testing data to a CSV file without the index column
        logger.debug('Train and test data saved to %s', raw_data_path)  # Log a debug message indicating successful saving
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)  # Log an error message if an exception occurs
        raise  # Re-raise the exception

def main():
    try:
        test_size = 0.21  # Set the proportion of the dataset to include in the test split (20%)
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'  # Define the URL of the raw data
        df = load_data(data_url=data_path)  # Call load_data function to fetch the data
        final_df = preprocess_data(df)  # Call preprocess_data function to rename columns
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)  # Split the data into training and testing sets with a fixed random state for reproducibility
        save_data(train_data, test_data, data_path='./data')  # Call save_data function to save the split datasets locally
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)  # Log an error if the overall process fails
        print(f"Error: {e}")  # Print the error message to the console

if __name__ == '__main__':  # Check if the script is being run directly (not imported as a module)
    main()  # Execute the main function
