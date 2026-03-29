import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

import numpy as np

df = pd.read_csv('data/BookCrossingThemes.csv',sep=None, engine='python')
#print(df.head())
print(df.columns)

#df['category'].unique()
#print(df['category'].nunique())
#print(df['Theme'].nunique())

#this part of the code prints the dataframe of the light readers, I created it so I can check easly the ID of
#a light reader in the input below
df_user = df.groupby('User-ID').agg({'Book-Title':list, 'category':list,'Book-Rating':list}).reset_index()
print(df_user.head())
df_light_reader = df_user[df_user['Book-Title'].apply(len)<=5]
print(df_light_reader.head())

user_id = int(input("Enter user ID: "))
user_data = df[df['User-ID'] == user_id]

if len(user_data)<= 5:
    recommanded_category = user_data.sort_values('Book-Rating',ascending=False).iloc[0]['category']
    #print(recommanded_category)
    df_books = df.groupby('Book-Title').agg({'Book-Rating':'mean','category':list}).reset_index()
    #print(df_books.head())
    #df_books[df_books['category']==recommanded_category]
    print(df_books[df_books['category'].apply(lambda x: recommanded_category in x)].sort_values('Book-Rating',ascending=False).iloc[0]['Book-Title'])
