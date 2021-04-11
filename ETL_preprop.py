# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine

def load_and_clean_datasets(messages_dir,categories_dir):
    # load messages dataset
    messages = pd.read_csv(messages_dir)

    # load categories dataset
    categories = pd.read_csv(categories_dir)
    # Merge the two
    df = messages.merge(categories,left_on='id',right_on='id')
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x.split('-')[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(columns=['categories'],inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df
    
def load_data(df,filename):
    # Save cleaned df to database
    engine = create_engine('sqlite:///'+ filename)
    df.to_sql('messages', engine, index=False)
    
def main():
    messages_dir, categories_dir, filename = sys.argv[1:]
    df = load_and_clean_datasets(messages_dir,categories_dir)        
    load_data(df,filename)
    

if __name__ == '__main__':
    main()

