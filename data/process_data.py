import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """   
    Description:    
        读取csv文件并合并。
    Input:
        messages_filepath：信息文件名。
        categories_filepath：标签分类文件名。
    Output:
        合并后的DataFrame。
    """
    #load messages dataset
    messages = pd.read_csv(messages_filepath)

    #load categories dataset
    categories = pd.read_csv(categories_filepath)

    #merge dataset to the 'df'
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Description:    
        清洗数据。
    Input:
        df：待清洗的DataFrame。
    Output:
        清洗后的DataFrame。
    """
    # Step 1:
    #create a dataframe of the 36 individual category columns

    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        #Elements in the split lists can be accessed using get or [] notation
        categories[column] = categories[column].str.split('-').str[1] 
        
        # convert column from string to numeric
        categories[column] = categories[column].astype("int")

    """
    Step 2:
    """
    # drop the original categories column from `df`
    df = df.drop(columns='categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    # TODO 11
    # 在这里进行合并，应该使用join，
    # 在索引或键列上将列与其他DataFrame连接起来。通过传递一个列表，一次有效地通过索引连接多个DataFrame对象。
    df = pd.concat([df,categories], axis=1)

    # check number of duplicates
    if len([i for i in df.duplicated() if i]) > 1:
        # drop duplicatesss
        df = df.drop_duplicates(subset=None, keep='first', inplace=False)

    return df


def save_data(df, database_filename):
    """
    Description:    
        保存DataFrame数据到Sqlite3数据库。
    Input:
        df：清洗后的DataFrame。
        database_filename：数据库文件名。
    Output:
        None。
    """
    #save the data to the database
    engine = create_engine('sqlite:///' + database_filename)

    df.to_sql('DisasterResponsePipeline_table', engine, index=False)  


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