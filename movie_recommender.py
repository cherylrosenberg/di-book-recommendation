import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

import numpy as np
from scipy.stats import pearsonr

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
    #Cold Start: just giving the best book according to the user's favorite category
    recommanded_category = user_data.sort_values('Book-Rating',ascending=False).iloc[0]['category']
    #print(recommanded_category)
    df_books = df.groupby('Book-Title').agg({'Book-Rating':'mean','category':list}).reset_index()
    #print(df_books.head())
    #df_books[df_books['category']==recommanded_category]
    recommanded_book = df_books[df_books['category'].apply(lambda x: recommanded_category in x)].sort_values('Book-Rating',ascending=False).iloc[0]['Book-Ti11tle']

if 5<len(user_data)<=20:
    #Warm Start: we no longer look at categories, we compare users to find the most similar ones (users who
    #give roughly the same ratings to the same books) in order to recommand a book)
    
    #TO DO: create a new df without the light readers

    users = df['User-ID'].values
    books = df['Book-Title'].values
    ratings = df['Book-Rating'].values

    #matrix : lines are users, columns are books
    unique_users = np.unique(users)
    unique_books = np.unique(books)

    user_to_index = {u: i for i, u in enumerate(unique_users)}
    book_to_index = {b: i for i, b in enumerate(unique_books)}

    matrix = np.zeros((len(unique_users), len(unique_books)))
    for u, b, r in zip(users, books, ratings):
        i = user_to_index[u]
        j = book_to_index[b]
        matrix[i, j] = r
    
    #we evaluate similarities with pearson test
    pearsonr_similarities = []

    for i in len(matrix):
        if i != user_to_index[user_id]:
            corr,p_value = pearsonr(matrix[i],matrix[user_to_index[user_id]])
            pearsonr_similarities.append((i,corr))

    pearsonr_similarities = pearsonr_similarities.sort(lambda x: x[1],reverse=True)
    j= 0
    similar_user = pearsonr_similarities[j][0]
    i = 0

    #we have the best similar user, we will now define the recommanded book
    while True:
        if i == len(unique_books): 
            #we have look at all the books of the actual similar user but we did not find a 
            #match for a recommended book because he has read the exactly same books for example
            #so we change similar_user
            i = 0
            j+=1
            similar_user = pearsonr_similarities[j][0]
            continue

        elif j >= len(pearsonr_similarities):
            recommended_book = None
            break
   
        elif matrix[user_to_index[user_id]][i] == 0 and matrix[similar_user][i] >= 8:
            recommanded_book = unique_books[i]
            break

        i+=1
        continue













    


     