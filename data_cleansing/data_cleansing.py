import pandas as pd 
import numpy as np
import html
import datetime
import ast
import re
import os 

book_ratings_raw = pd.read_csv('../data/BookCrossingThemes.csv', sep = ';')

book_ratings = book_ratings_raw.copy()

# 1. Count the number of duplicate user-item pairs. This will confirm if there are any duplicate reviews for the same book by the same user
duplicate_count = book_ratings.duplicated(subset=['User-ID', 'ISBN']).sum()
print(f"Number of duplicate (User, ISBN) pairs: {duplicate_count}")

# Create a mapping of Title -> First ISBN found
title_to_isbn = book_ratings.groupby('Book-Title')['ISBN'].first().to_dict()

# Map all rows to use the 'Master ISBN' for their title
book_ratings['Master-ISBN'] = book_ratings['Book-Title'].map(title_to_isbn)

# Check the new count (should be 1 ISBN per Title)
new_counts = book_ratings.groupby('Book-Title')['Master-ISBN'].nunique()
print(f"Unique Master-ISBNs per title: {new_counts.max()}")

duplicate_count = book_ratings.duplicated(subset=['User-ID', 'Master-ISBN']).sum()
print(f"Number of duplicate (User, ISBN) pairs: {duplicate_count}")

# Drop duplicates based on User-ID and the new Master-ISBN
book_ratings = book_ratings.drop_duplicates(subset=['User-ID', 'Master-ISBN'], keep='first')

# Force every value in Master-ISBN to be a string and strip any accidental whitespace
book_ratings['Master-ISBN'] = book_ratings['Master-ISBN'].astype(str).str.strip()

user_counts = book_ratings['User-ID'].value_counts()

print("User Rating Stats:")
print(user_counts.describe())

# Columns to clean
text_cols = ['Book-Title', 'Book-Author', 'Publisher']

for col in text_cols:
    # 1. Ensure the column is string type
    # 2. Unescape HTML entities (e.g., &amp; -> &)
    # 3. Trim leading/trailing whitespace
    book_ratings[col] = (
        book_ratings[col]
        .astype(str)
        .apply(html.unescape)
        .str.strip()
    )

print("Text columns cleaned: HTML entities decoded and whitespace trimmed.")

# Apply Title Case to unify capitalization
for col in ['Book-Title', 'Book-Author', 'Publisher']:
    book_ratings[col] = book_ratings[col].str.title()

print("Columns converted to Title Case.")

# Identify all rows that have a duplicate (User-ID, Book-Title) pair
duplicates_view = book_ratings[book_ratings.duplicated(subset=['User-ID', 'Book-Title'], keep=False)]

# Sort them so the pairs are next to each other
print(duplicates_view.sort_values(by=['User-ID', 'Book-Title']).head(10))

# Group by User-ID and Book-Title
# Take the mean of 'Book-Rating' and keep the first occurrence for everything else
aggregation_rules = {col: 'first' for col in book_ratings.columns if col not in ['User-ID', 'Book-Title', 'Book-Rating']}
aggregation_rules['Book-Rating'] = 'mean'

book_ratings = book_ratings.groupby(['User-ID', 'Book-Title'], as_index=False).agg(aggregation_rules)

# Optional: Round the rating if you want to keep them as integers
book_ratings['Book-Rating'] = book_ratings['Book-Rating'].round().astype(int)

print("Duplicates merged using the mean rating.")

# 1. Define the range of valid years
# Any year above 'current_year + 1' is likely an error (unless it's a pre-order)
current_year = datetime.datetime.now().year

# 2. Identify invalid rows
# Checking for 0, negative numbers, or future years
invalid_years = book_ratings[
    (book_ratings['Year-Of-Publication'] <= 0)  | 
    (book_ratings['Year-Of-Publication'] > current_year + 1)
]

# 3. View the results
print(f"Total invalid years found: {len(invalid_years)}")
if not invalid_years.empty:
    print(invalid_years[['Book-Title', 'Year-Of-Publication']].value_counts().head(10))

book_ratings['Year-Of-Publication'] = book_ratings['Year-Of-Publication'].replace(0, np.nan)

# 2. Fill NaNs with the median year for that author
book_ratings['Year-Of-Publication'] = book_ratings.groupby('Book-Author')['Year-Of-Publication'].transform(
    lambda x: x.fillna(x.median())
)

# 1. Identify the authors who have a year of 0
# authors_with_nan = book_ratings[book_ratings['Year-Of-Publication'].isna()]['Book-Author'].unique()

# # 2. Check how many books those specific authors have in the ENTIRE dataset
# author_counts = book_ratings[book_ratings['Book-Author'].isin(authors_with_nan)].groupby('Book-Author').size()

# # 3. Filter for authors who have MORE than just the one book with year 0
# authors_with_data = author_counts[author_counts > 1]

# print(f"Total authors with at least one Year-0 book: {len(authors_with_nan)}")
# print(f"Authors you can successfully 'fix' using their other books: {len(authors_with_data)}")
# print(f"Authors with NO other data (will need the global median): {len(authors_with_nan) - len(authors_with_data)}")

# 2. Final check: ensure there are no NaNs left
# remaining_nans = book_ratings['Year-Of-Publication'].isna().sum()
# print(f"Remaining NaNs in Year-Of-Publication: {remaining_nans}")

# 1. Fill any remaining NaNs with the global median of the entire column
global_median = book_ratings['Year-Of-Publication'].median()
book_ratings['Year-Of-Publication'] = book_ratings['Year-Of-Publication'].fillna(global_median)

# 2. Now it is safe to convert to integer
book_ratings['Year-Of-Publication'] = book_ratings['Year-Of-Publication'].astype(int)

print(f"Remaining NaNs: {book_ratings['Year-Of-Publication'].isna().sum()}")
print(f"New Column Type: {book_ratings['Year-Of-Publication'].dtype}")

def normalize_to_title_case(cat):
    # Handle empty, null, or 'nan'
    if pd.isna(cat) or str(cat).lower() == 'nan' or str(cat).strip() == '':
        return []
    
    val = str(cat).strip()
    
    # 1. Handle string-represented lists like "['science fiction', 'DRAMA']"
    if val.startswith('[') and val.endswith(']'):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                # .title() turns "science fiction" into "Science Fiction"
                return [str(item).strip().title() for item in parsed if item]
        except (ValueError, SyntaxError):
            # Fallback for messy strings that look like lists
            val = re.sub(r"[\[\]\'\"]", "", val)
    
    # 2. Split by commas and title-case each element
    tags = [t.strip().title() for t in re.split(r',', val) if t.strip()]
    return tags

# Apply the normalization
book_ratings['category_list'] = book_ratings['category'].apply(normalize_to_title_case)

book_ratings['primary_category'] = book_ratings['category_list'].apply(
    lambda lst: lst[0] if isinstance(lst, list) and len(lst) > 0 else np.nan
)
book_ratings['primary_category'] = book_ratings['primary_category'].fillna(book_ratings['Theme'])

# # Count rows where the list length is greater than 1
# multi_tag_count = book_ratings['category_list'].apply(lambda x: len(x) > 1).sum()

# # Calculate the percentage for context
# total_rows = len(book_ratings)
# percentage = (multi_tag_count / total_rows) * 100

# print(f"Rows with multiple categories: {multi_tag_count}")
# print(f"Percentage of dataset: {percentage:.2f}%")

# 1. Count exact nulls
null_age_count = book_ratings['Age'].isna().sum()
print(f"Total Nulls in Age: {null_age_count}")

# Replace outliers with NaN so they can be filled by the median later
book_ratings.loc[(book_ratings['Age'] < 5) | (book_ratings['Age'] > 100), 'Age'] = np.nan

# Group by Location and calculate median, standard deviation, and count
location_stats = book_ratings.groupby('Location')['Age'].agg(['median', 'std', 'count'])

# 1. Fill NaNs using the median age for each specific Location
book_ratings['Age'] = book_ratings.groupby('Location')['Age'].transform(
    lambda x: x.fillna(x.median())
)

# 2. Safety Fallback: For users in locations with NO age data at all, use the global median
global_median = book_ratings['Age'].median()
book_ratings['Age'] = book_ratings['Age'].fillna(global_median)

# 3. Final cleanup: Round to integer (ages aren't decimals)
book_ratings['Age'] = book_ratings['Age'].round().astype(int)

user_counts = book_ratings['User-ID'].value_counts()

# Define the path to the data folder
output_path = os.path.join('..', 'data', 'BookCrossingThemes_Updated.csv')

# Save the dataframe
book_ratings.to_csv(output_path, index=False)

print(f"File successfully saved to: {output_path}")