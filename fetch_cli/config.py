BASE_URL = "https://arctic-shift.photon-reddit.com"
VALID_COMMENT_API_FIELDS = "id,created_utc,author,score,body,parent_id,link_id,subreddit"
VALID_POST_API_FIELDS = (
    "id,created_utc,title,selftext,score,num_comments,author,subreddit,url,link_flair_text,over_18"
)
DEFAULT_POST_COLUMNS = [
    "id",
    "created_utc",
    "created_datetime",
    "title",
    "selftext",
    "score",
    "upvote_ratio",
    "num_comments",
    "author",
    "subreddit",
    "url",
    "permalink",
    "link_flair_text",
    "is_self",
    "over_18",
]