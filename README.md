# Reddit Sentiment Analysis Data Collection

Python utilities for retrieving Reddit posts and their comments from the Arctic Shift API. The collected data can be used as input to a sentiment-analysis workflow.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Edit the example in `main.py` to choose a subreddit, date range, and ranking options, then run:

```bash
python main.py
```

The script writes post and comment datasets as CSV files in the project directory. Those generated datasets, the local virtual environment, and the bundled Arctic Shift checkout are deliberately excluded from version control.

## Connect to GitHub

After creating an empty GitHub repository, add it as the `origin` remote and push the `main` branch:

```bash
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```
