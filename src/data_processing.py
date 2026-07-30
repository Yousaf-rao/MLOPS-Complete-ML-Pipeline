# Import the os module to interact with the operating system (Example: os.makedirs('logs'))
import os
# Import the logging module to record events during execution (Example: logger.info('Started processing'))
import logging
# Import pandas for data manipulation and analysis (Example: df = pd.read_csv('data.csv'))
import pandas as pd
# Import LabelEncoder to convert categorical labels into numeric values (Example: encoder.fit_transform(['spam', 'ham']))
from sklearn.preprocessing import LabelEncoder
# Import PorterStemmer to reduce words to their root form (Example: stemmer.stem('running') -> 'run')
from nltk.stem.porter import PorterStemmer
# Import stopwords to filter out common uninformative words (Example: 'the', 'is', 'in' are removed)
from nltk.corpus import stopwords
# Import string module to access common string operations and constants (Example: string.punctuation contains '!"#$%&')
import string
# Import nltk for natural language processing tasks (Example: nltk.word_tokenize('Hello world'))
import nltk
# Download the 'stopwords' dataset quietly without printing output (Example: downloads words like 'and', 'the')
nltk.download('stopwords', quiet=True)
# Download the 'punkt' tokenizer models quietly (Example: used to split sentences into words)
nltk.download('punkt', quiet=True)
# Download the 'punkt_tab' models quietly (Example: additional data required by the tokenizer)
nltk.download('punkt_tab', quiet=True)

# Create a set of English stopwords at the module level for fast lookup (Example: STOP_WORDS = {'a', 'an', 'the', ...})
STOP_WORDS = set(stopwords.words('english'))
# Initialize the PorterStemmer once at the module level for efficiency (Example: PS.stem('cats') -> 'cat')
PS = PorterStemmer()

# Define the directory name where logs will be stored (Example: log_dir = 'logs')
log_dir = 'logs'
# Create the log directory if it doesn't exist, ignore if it already exists (Example: creates folder './logs')
os.makedirs(log_dir, exist_ok=True)

# Create a logger object named 'data_preprocessing' to record specific messages (Example: logger = logging.getLogger('my_app'))
logger = logging.getLogger('data_preprocessing')
# Set the logger's threshold to DEBUG to capture detailed diagnostic information (Example: logger.setLevel(logging.DEBUG))
logger.setLevel('DEBUG')

# Create a handler to print log messages to the console (Example: console_handler = logging.StreamHandler())
console_handler = logging.StreamHandler()
# Set the console handler's threshold to DEBUG so it outputs all debug messages (Example: console_handler.setLevel('DEBUG'))
console_handler.setLevel('DEBUG')

# Define the full file path for the log file (Example: log_file_path = 'logs/data_preprocessing.log')
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
# Create a handler to write log messages to the specified file (Example: file_handler = logging.FileHandler('app.log'))
file_handler = logging.FileHandler(log_file_path)
# Set the file handler's threshold to DEBUG so it writes all debug messages (Example: file_handler.setLevel('DEBUG'))
file_handler.setLevel('DEBUG')

# Define the format for log messages, including time, logger name, level, and message (Example: '2023-10-01 12:00:00 - data_preprocessing - INFO - Done')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Apply the defined formatter to the console handler (Example: console_handler.setFormatter(formatter))
console_handler.setFormatter(formatter)
# Apply the defined formatter to the file handler (Example: file_handler.setFormatter(formatter))
file_handler.setFormatter(formatter)

# Attach the console handler to the logger so messages are printed (Example: logger.addHandler(console_handler))
logger.addHandler(console_handler)
# Attach the file handler to the logger so messages are saved to the file (Example: logger.addHandler(file_handler))
logger.addHandler(file_handler)

# Define a function to process and clean input text strings (Example: def transform_text('Hello World!'):)
def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    # Convert the entire text to lowercase for uniform processing (Example: 'Hello' becomes 'hello')
    text = text.lower()
    # Split the text into a list of individual word tokens (Example: 'hello world' becomes ['hello', 'world'])
    text = nltk.word_tokenize(text)
    # Keep only tokens that contain alphabetic letters or numbers, filtering out symbols (Example: ['hello', '!', '123'] becomes ['hello', '123'])
    text = [word for word in text if word.isalnum()]
    # Keep only tokens that are not in the STOP_WORDS set and not punctuation (Example: ['the', 'cat', ','] becomes ['cat'])
    text = [word for word in text if word not in STOP_WORDS and word not in string.punctuation]
    # Reduce each word to its root stem using PorterStemmer (Example: ['running', 'dogs'] becomes ['run', 'dog'])
    text = [PS.stem(word) for word in text]
    # Combine the list of stemmed words back into a single string separated by spaces (Example: ['run', 'dog'] becomes 'run dog')
    result = " ".join(text)
    # Return the final cleaned and stemmed string (Example: return 'run dog')
    return result

# Define a function to preprocess a pandas DataFrame containing text and labels (Example: preprocess_df(df, 'sms', 'label'))
def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    # Begin a try block to handle any potential errors during preprocessing (Example: try:)
    try:
        # Log a debug message indicating the start of DataFrame preprocessing (Example: logger.debug('Starting...'))
        logger.debug('Starting preprocessing for DataFrame')
        # Initialize a LabelEncoder object to transform textual labels into integers (Example: encoder = LabelEncoder())
        encoder = LabelEncoder()
        # Fit the encoder to the target column and transform 'ham'/'spam' to 0/1 (Example: df['target'] = [0, 1, 0])
        df[target_column] = encoder.fit_transform(df[target_column])
        # Log a debug message confirming the target column was encoded (Example: logger.debug('Encoded'))
        logger.debug('Target column encoded')

        # Drop any duplicate rows from the DataFrame, keeping only the first occurrence (Example: df.drop_duplicates())
        df = df.drop_duplicates(keep='first')
        # Log a debug message confirming that duplicate rows were removed (Example: logger.debug('Removed duplicates'))
        logger.debug('Duplicates removed')
        
        # Apply the transform_text function to each entry in the text column (Example: df['text'] = df['text'].apply(cleaner))
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        # Log a debug message confirming the text column was successfully transformed (Example: logger.debug('Text transformed'))
        logger.debug('Text column transformed')
        # Return the fully preprocessed DataFrame (Example: return processed_df)
        return df
    
    # Catch a KeyError if the specified text or target columns are not found in the DataFrame (Example: except KeyError as e:)
    except KeyError as e:
        # Log an error message with the missing column name (Example: logger.error('Missing column: text'))
        logger.error('Column not found: %s', e)
        # Re-raise the KeyError to halt execution and signal the failure (Example: raise)
        raise
    # Catch any other unexpected exceptions that occur during text normalization (Example: except Exception as e:)
    except Exception as e:
        # Log an error message containing the exception details (Example: logger.error('Error occurred'))
        logger.error('Error during text normalization: %s', e)
        # Re-raise the exception to halt execution (Example: raise)
        raise

# Define the main function that coordinates the loading, processing, and saving of data (Example: def main():)
def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    # Begin a try block to handle potential errors during the main execution flow (Example: try:)
    try:
        # Read the raw training data from a CSV file into a pandas DataFrame (Example: pd.read_csv('train.csv'))
        train_data = pd.read_csv('./data/raw/train.csv')
        # Read the raw testing data from a CSV file into a pandas DataFrame (Example: pd.read_csv('test.csv'))
        test_data = pd.read_csv('./data/raw/test.csv')
        # Log a debug message indicating that both datasets were loaded successfully (Example: logger.debug('Loaded data'))
        logger.debug('Data loaded properly')

        # Process the training DataFrame by encoding labels and cleaning text (Example: preprocess_df(train))
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        # Process the testing DataFrame by encoding labels and cleaning text (Example: preprocess_df(test))
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Create a variable storing the path to the interim processed data directory (Example: path = './data/interim')
        data_path = os.path.join("./data", "interim")
        # Create the interim directory if it does not exist, ignoring if it does (Example: os.makedirs('./data/interim'))
        os.makedirs(data_path, exist_ok=True)
        
        # Save the processed training DataFrame to a new CSV file without row indices (Example: df.to_csv('out.csv', index=False))
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        # Save the processed testing DataFrame to a new CSV file without row indices (Example: df.to_csv('out2.csv', index=False))
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        # Log a debug message indicating where the processed data files were saved (Example: logger.debug('Saved to dir'))
        logger.debug('Processed data saved to %s', data_path)
    # Catch a FileNotFoundError if the raw input CSV files cannot be located (Example: except FileNotFoundError:)
    except FileNotFoundError as e:
        # Log an error message with the file missing exception details (Example: logger.error('File missing'))
        logger.error('File not found: %s', e)
        # Re-raise the FileNotFoundError to halt execution (Example: raise)
        raise
    # Catch an EmptyDataError if the loaded CSV file is completely empty (Example: except pd.errors.EmptyDataError:)
    except pd.errors.EmptyDataError as e:
        # Log an error message detailing that no data was found in the CSV (Example: logger.error('CSV empty'))
        logger.error('No data: %s', e)
        # Re-raise the EmptyDataError to halt execution (Example: raise)
        raise
    # Catch any other generic exceptions that occur during the main process (Example: except Exception as e:)
    except Exception as e:
        # Log an error message describing the failure during data transformation (Example: logger.error('Process failed'))
        logger.error('Failed to complete the data transformation process: %s', e)
        # Print a fallback error message directly to standard output (Example: print('Error: ...'))
        print(f"Error: {e}")
        # Re-raise the exception to abort the program (Example: raise)
        raise

# Check if this Python script is being run directly as the main program (Example: if __name__ == '__main__':)
if __name__ == '__main__':
    # Execute the main function to start the preprocessing pipeline (Example: main())
    main()