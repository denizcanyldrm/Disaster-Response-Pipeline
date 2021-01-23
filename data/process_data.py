import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Read two file paths into dataframes

    Args:
        messages_filepath (str): file path contains messages information
        categories_filepath (str): file path contains categories information 
    
    Returns:
        dataframe: a dataframe with messages and categories dataframes combined
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories, how='outer', on=['id'])


def clean_data(df):
    """Clean data in the dataframe. 

    - Create different columns for each message categories
    - Drop original category column of dataframe
    - Format messages column as 'str'
    - Format each category column as 'int32'
    
    Args:
        df (dataframe): a dataframe
    
    Returns:
        dataframe: a cleansed dataframe 
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract a list of new column names for categories by using this row
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    # For example, 'related-0' becomes '0', 'related-1' becomes '1'. 
    # Convert the string to a numeric value.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
  
    # drop the original categories column from 'df'
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
                 
    # drop duplicates
    df.drop_duplicates(inplace=True)
    df = df[df['related'] != 2]

    return df
    

def save_data(df, database_filename):
    """Save dataframe into sql database
    
    Args:
        df (dataframe): a dataframe
        database_filename (str): file name for database
    
    Returns:
        None
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    tableName = database_filename[5:]
    df.to_sql(tableName[:-3], engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()