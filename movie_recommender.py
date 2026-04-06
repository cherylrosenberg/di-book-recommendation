import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

import numpy as np
from scipy.stats import pearsonr
from scipy.cluster.vq import kmeans, vq, whiten

df = pd.read_csv('data/BookCrossingThemes_Updated.csv',sep=None, engine='python')


#########THIS PART FITS BETTER IN THE CLEANSING FILE##########
bins = [0, 12, 17, 59, 100]
labels = ['Children', 'Young Adult', 'Adult', 'Senior']
df['Age-Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
df['Country'] = df['Location'].str.split(',').str[-1].str.strip()


def build_rating_matrix(df: pd.DataFrame):
    # Return (matrix, unique_users, unique_books, user_to_index, book_to_index)
    users  = df['User-ID'].values
    books  = df['Book-Title'].values
    ratings = df['Book-Rating'].values
 
    unique_users = np.unique(users)
    unique_books = np.unique(books)
 
    user_to_index = {u: i for i, u in enumerate(unique_users)}
    book_to_index = {b: i for i, b in enumerate(unique_books)}
 
    matrix = np.zeros((len(unique_users), len(unique_books)))
    for u, b, r in zip(users, books, ratings):
        matrix[user_to_index[u], book_to_index[b]] = r
 
    return matrix, unique_users, unique_books, user_to_index, book_to_index


matrix, unique_users, unique_books, user_to_index, book_to_index = build_rating_matrix(df)


def recommender_for_light_user(user_data: pd.DataFrame) -> list:
    # Cold start — recommend the top-rated unread book matching the user's favourite category AND theme.
    # If no book matches both, fall back to category only.
    recommended_books = []
 
    best_rated_row = user_data.sort_values('Book-Rating', ascending=False).iloc[0]
    recommended_category = best_rated_row['primary_category']
    recommended_theme    = best_rated_row['Theme']
 
    df_books = (
        df.groupby('Book-Title')
        .agg({'Book-Rating': 'mean', 'primary_category': list, 'Theme': list})
        .reset_index()
    )
 
    # First try: category AND theme
    candidate_df = (
        df_books[
            df_books['primary_category'].apply(lambda x: recommended_category in x) &
            df_books['Theme'].apply(lambda x: recommended_theme in x)
        ]
        .sort_values('Book-Rating', ascending=False)
    )
 
    # Fallback: category only
    if candidate_df.empty:
        candidate_df = (
            df_books[df_books['primary_category'].apply(lambda x: recommended_category in x)]
            .sort_values('Book-Rating', ascending=False)
        )
 
    already_read = set(user_data['Book-Title'])
    i = 0
    while i < len(candidate_df):
        book = candidate_df.iloc[i]['Book-Title']
        if book in already_read:
            i += 1
            continue
        recommended_books.append(book)
        break

    return recommended_books
        


def recommender_for_medium_user(user_data):
    #between 5 and 20 books
    #Warm Start: we no longer look at categories, we compare users to find the most similar ones (users who
    #give roughly the same ratings to the same books) in order to recommand a book)
    #Similarities (User-based)
    recommended_books = []
    already_read = set(user_data['Book-Title'])

    user_id      = user_data['User-ID'].iloc[0]
    target_index = user_to_index.get(user_id)
    if target_index is None:
        return recommended_books
    
    target_vector = matrix[target_index]
    similarities  = []

    for i in range(len(matrix)):
        if i == target_index:
            continue

        # compute the corrolation on books both users have rated
        mask = (matrix[i] > 0) & (target_vector > 0)
        if mask.sum() < 3:          # need at least 3 co-rated books
            continue

        corr, _ = pearsonr(matrix[i][mask], target_vector[mask])
        if not np.isnan(corr):
            similarities.append((i, corr))

    similarities.sort(key=lambda x: x[1], reverse=True)

    j = 0
    i = 0
    while True:
        if j >= len(similarities):
            break
        similar_user = similarities[j][0]
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
    # Heavy reader : more than 20 books
    #collabortive filtering, item based : someone that likes this book typically also like this other book
    recommended_books = []
    already_read = set(user_data['Book-Title'])

    user_id      = user_data['User-ID'].iloc[0]
    target_index = user_to_index.get(user_id)
    if target_index is None:
        return recommended_books

    user_ratings = matrix[target_index]
    liked_book_indices = np.where(user_ratings >= 8)[0]

    #Compute the corrolation per liked book
    for book_index in liked_book_indices:
        book_vector = matrix[:, book_index]
        best_corr   = -1
        best_idx    = None
 
        for j in range(matrix.shape[1]):
            if j == book_index:
                continue
 
            # only correlate over users who rated both books
            mask = (book_vector > 0) & (matrix[:, j] > 0)
            if mask.sum() < 3:
                continue
 
            corr, _ = pearsonr(book_vector[mask], matrix[:, j][mask])
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_idx  = j
 
        if best_idx is not None and best_corr >= 0.7:
            candidate_book = unique_books[best_idx]
            if candidate_book not in already_read and candidate_book not in recommended_books:
                recommended_books.append(candidate_book)
                break
 
    return recommended_books


#Clustering : category, age and location

def build_cluster_assignments(df: pd.DataFrame, k: int = 6):
    # Return (user_profile_index, cluster_labels) computed once
    user_profile = df.pivot_table(
        index='User-ID',
        columns='primary_category',
        values='Book-Rating',
        aggfunc='mean',
        fill_value=0
    )
 
    age_df     = df.groupby('User-ID')['Age-Group'].first()
    country_df = df.groupby('User-ID')['Country'].first()
    # Theme favori = le thème du livre le mieux noté par l'user
    theme_df   = df.groupby('User-ID').apply(lambda x: x.loc[x['Book-Rating'].idxmax(), 'Theme'])
 
    user_profile['Age-Group'] = age_df
    user_profile = pd.get_dummies(user_profile, columns=['Age-Group'])
 
    user_profile['Country'] = country_df
    user_profile = pd.get_dummies(user_profile, columns=['Country'])
 
    user_profile['Theme'] = theme_df
    user_profile = pd.get_dummies(user_profile, columns=['Theme'])
 
    numpy_matrix       = user_profile.to_numpy(dtype=float)
    numpy_matrix_white = whiten(numpy_matrix)
 
    np.random.seed(42)                                  # reproducibility
    centroids, _       = kmeans(numpy_matrix_white, k)
    cluster_labels, _  = vq(numpy_matrix_white, centroids)
 
    cluster_users    = user_profile.index.to_numpy()
    user_to_cluster  = {user: cluster_labels[i] for i, user in enumerate(cluster_users)}
 
    return cluster_users, cluster_labels, user_to_cluster


cluster_users, cluster_labels, user_to_cluster = build_cluster_assignments(df)


def recommender_from_cluster(user_id, n: int = 2) -> list:
    # Recommend top-rated books from the user's demographic cluster.
    user_data    = df[df['User-ID'] == user_id]
    already_read = set(user_data['Book-Title'])
    recommended_books = []
 
    if user_id not in user_to_cluster:
        return recommended_books
 
    target_cluster     = user_to_cluster[user_id]
    same_cluster_users = cluster_users[cluster_labels == target_cluster]
 
    cluster_df = df[df['User-ID'].isin(same_cluster_users)]
    liked_by_cluster = (
        cluster_df.groupby('Book-Title')['Book-Rating']
        .mean()
        .sort_values(ascending=False)
    )
 
    for book in liked_by_cluster.index:
        if book not in already_read:
            recommended_books.append(book)
            if len(recommended_books) == n:
                break
 
    return recommended_books



def recommend_books(user_id):
    user_data = df[df['User-ID'] == user_id]
 
    if user_data.empty:
        return ["User ID not found"]
 
    final_recommendations = []
 
    if len(user_data) <= 5:
        final_recommendations.extend(recommender_for_light_user(user_data))
 
    elif 5 < len(user_data) <= 20:
        final_recommendations.extend(recommender_for_medium_user(user_data))
 
    else:
        final_recommendations.extend(recommender_for_high_users(user_data))
 
    # calling this function if and only if our list isn't long enough
    if len(final_recommendations) < 3:
        n_missing = 3 - len(final_recommendations)
        cluster_books = recommender_from_cluster(user_id, n=n_missing)
        for book in cluster_books:
            if book not in final_recommendations:
                final_recommendations.append(book)
 
    return final_recommendations[:3]


