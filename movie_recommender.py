import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

import numpy as np
from scipy.stats import pearsonr
from scipy.cluster.vq import kmeans, vq, whiten

df = pd.read_csv('data/BookCrossingThemes_Updated.csv',sep=None, engine='python')
#print(df.head())


#df['category'].unique()
#print(df['category'].nunique())
#print(df['Theme'].nunique())

#this part of the code prints the dataframe of the light readers, I created it so I can check easly the ID of
#a light reader in the input below
#df_user = df.groupby('User-ID').agg({'Book-Title':list, 'primary_category':list,'Book-Rating':list}).reset_index()
#print(df_user.head())
#df_light_reader = df_user[df_user['Book-Title'].apply(len)<=5]
#print(df_light_reader.head())


#########THIS PART FITS BETTER IN THE CLEANSING FILE##########
bins = [0, 12, 17, 59, 100]
labels = ['Children', 'Young Adult', 'Adult', 'Senior']
df['Age-Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
df['Country'] = df['Location'].str.split(',').str[-1].str.strip()


print(df.columns)





def recommender_for_light_user(user_data):
#if len(user_data)<= 5:
    #Cold Start: just giving the best book according to the user's favorite category
    recommanded_books = []
    recommanded_category = user_data.sort_values('Book-Rating',ascending=False).iloc[0]['primary_category'] 
    #print(recommanded_category)
    df_books = df.groupby('Book-Title').agg({'Book-Rating':'mean','primary_category':list}).reset_index()
    #print(df_books.head())
    #df_books[df_books['category']==recommanded_category]
    
    #we need to check that the recommended book is not in the list of the books that the user has already read
    i = 0
    recommanded_book_df = df_books[df_books['primary_category'].apply(lambda x: recommanded_category in x)].sort_values('Book-Rating',ascending=False)
    while i < len(recommanded_book_df):
        book = recommanded_book_df.iloc[i]['Book-Title']
        if book in user_data['Book-Title'].values:
            i+=1
            continue
        
        recommanded_books.append(book)
        # return recommanded_book #TO DO SPELLING
        break

    return recommanded_books
        


def recommender_for_medium_user(user_data):
    #if 5<len(user_data)<=20:
    #Warm Start: we no longer look at categories, we compare users to find the most similar ones (users who
    #give roughly the same ratings to the same books) in order to recommand a book)
    #Similarities (User-based)
    recommended_books = []
    already_read = set(user_data['Book-Title'])

    users = df['User-ID'].values
    books = df['Book-Title'].values
    ratings = df['Book-Rating'].values

    unique_users = np.unique(users)
    unique_books = np.unique(books)

    user_to_index = {u: i for i, u in enumerate(unique_users)}
    book_to_index = {b: i for i, b in enumerate(unique_books)}

    matrix = np.zeros((len(unique_users), len(unique_books)))
    for u, b, r in zip(users, books, ratings):
        i = user_to_index[u]
        j = book_to_index[b]
        matrix[i, j] = r

    pearsonr_similarities = []

    target_index = user_to_index[user_id]

    for i in range(len(matrix)):
        if i != target_index:
            corr, p_value = pearsonr(matrix[i], matrix[target_index])
            if not np.isnan(corr):
                pearsonr_similarities.append((i, corr))

    pearsonr_similarities.sort(key=lambda x: x[1], reverse=True)

    j = 0
    i = 0

    while True:
        if j >= len(pearsonr_similarities):
            break

        similar_user = pearsonr_similarities[j][0]

        if i == len(unique_books):
            i = 0
            j += 1
            continue

        if matrix[target_index][i] == 0 and matrix[similar_user][i] >= 8:
            candidate_book = unique_books[i]
            if candidate_book not in already_read and candidate_book not in recommended_books:
                recommended_books.append(candidate_book)
                break

        i += 1

    return recommended_books



def recommender_for_high_users(user_data):
    #if 20<len(user_data): #collabortive filtering, item based : someone that likes this book typically also like this other book
    recommended_books = []
    already_read = set(user_data['Book-Title'])

    users = df['User-ID'].values
    books = df['Book-Title'].values
    ratings = df['Book-Rating'].values

    unique_users = np.unique(users)
    unique_books = np.unique(books)

    user_to_index = {u: i for i, u in enumerate(unique_users)}
    book_to_index = {b: i for i, b in enumerate(unique_books)}

    matrix = np.zeros((len(unique_users), len(unique_books)))
    for u, b, r in zip(users, books, ratings):
        i = user_to_index[u]
        j = book_to_index[b]
        matrix[i, j] = r

    target_index = user_to_index[user_id]
    user_ratings = matrix[target_index]
    liked_books = np.where(user_ratings >= 8)[0]

    corr_matrix = np.corrcoef(matrix.T)

    i = 0
    while i < len(liked_books):
        book_index = liked_books[i]
        corr_chosen_book = corr_matrix[book_index]
        sim_row = corr_chosen_book.copy()
        sim_row[book_index] = -1
        most_similar_index = np.argmax(sim_row)

        if corr_chosen_book[most_similar_index] >= 0.7:
            candidate_book = unique_books[most_similar_index]
            if candidate_book not in already_read and candidate_book not in recommended_books:
                recommended_books.append(candidate_book)
                break

        i += 1

    return recommended_books


#Clustering : category, age and location

def recommender_from_cluster(user_data, n=2):
    already_read = set(user_data['Book-Title'])
    recommended_books = []

    user_profile = df.pivot_table(
        index='User-ID',
        columns='primary_category',
        values='Book-Rating',
        aggfunc='mean',
        fill_value=0
    )

    age_df = df.groupby('User-ID')['Age-Group'].first()
    user_profile['Age-Group'] = age_df
    user_profile = pd.get_dummies(user_profile, columns=['Age-Group'])

    country_df = df.groupby('User-ID')['Country'].first()
    user_profile['Country'] = country_df
    user_profile = pd.get_dummies(user_profile, columns=['Country'])

    numpy_matrix = user_profile.to_numpy(dtype=float)
    numpy_matrix_white = whiten(numpy_matrix)

    k = 6
    centroids, distortion = kmeans(numpy_matrix_white, k)
    cluster_labels, distances = vq(numpy_matrix_white, centroids)

    cluster_users = user_profile.index.to_numpy()

    user_to_cluster = {
        user: cluster_labels[i]
        for i, user in enumerate(cluster_users)
    }

    if user_id not in user_to_cluster:
        return []

    target_cluster = user_to_cluster[user_id]
    same_cluster_users = cluster_users[cluster_labels == target_cluster]

    cluster_df = df[df['User-ID'].isin(same_cluster_users)]

    liked_by_cluster = (
        cluster_df.groupby('Book-Title')['Book-Rating']
        .mean()
        .sort_values(ascending=False)
    )

    for book in liked_by_cluster.index:
        if book not in already_read and book not in recommended_books:
            recommended_books.append(book)
        if len(recommended_books) == n:
            break

    return recommended_books







    


     