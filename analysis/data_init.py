import sqlite3
import pandas as pd

DATABASE = "data/socialtest.sqlite"
RUN_ID = "test-month-no-askreddit"

def get_comments(run_id: str): # -> pd.DataFrame
    with sqlite3.connect(DATABASE) as connection:
        comments = pd.read_sql_query(
            "SELECT * FROM comments WHERE run_id = ?",
            connection,
            params=(run_id,)
        )
    return comments
    
def get_posts(run_id: str): # -> pd.DataFrame
    with sqlite3.connect(DATABASE) as connection:
        posts = pd.read_sql_query(
            "SELECT * FROM posts WHERE run_id = ?",
            connection,
            params=(run_id,)
        )
        
    return posts

# change_indicies = comments.index[comments['run_id'] != 0]
# print(change_indicies)

# comments = comments[~(comments['run_id'] == 'test-month')]

# comments.drop(comments.index[:144], inplace=True)

# print(comments.head())