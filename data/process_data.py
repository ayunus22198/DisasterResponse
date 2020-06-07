import sys
import pandas as pd
from sqlalchemy import create_engine


def clean_categories(df):
    pass

'''
Description: Load data
'''
def load_data(messages_filepath, categories_filepath):
    df_msg = pd.read_csv(messages_filepath)
    df_ctgry = pd.read_csv(categories_filepath)
    categories = df_ctgry.drop(['id'], axis=1)
    frames = [df_msg, categories]
    df = pd.concat(frames, axis = 1)
    return df

'''
Description: Cleans data and renames columns, then concatenates to form one dataframe
'''
def clean_data(df):
    cats = df['categories'].str.split(";", expand = True)
    cols = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'] 
    mapping = {}
    for i in range(36):
        mapping[i] = cols[i]

    cats.rename(columns = mapping, inplace = True)
    
    for column in cats:
        # set each value to be the last character of the string
        cats[column] = cats[column][1][-1]
        #print(categories[column][1][-1])
        # convert column from string to numeric
        cats[column] = pd.to_numeric(cats[column])

    df = df.drop(['categories'], axis = 1)
    frames_second = [df, cats]
    df = pd.concat(frames_second, axis = 1)
    return df


'''
Description: Save database in file system
'''
def save_data(df, database_filename):
    df.drop_duplicates(inplace = True)
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disasters', engine, index=False)
    pass  


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