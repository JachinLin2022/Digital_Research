# Bitcoin Subreddit Scraper

This is a Reddit scraper that extracts posts from the Bitcoin subreddit using Python's PRAW library.

## Requirements

This project requires Python 3 and the PRAW (Python Reddit API Wrapper) library to be installed. You also need to create a Reddit API account and acquire your API credentials.

## Installation

1. Clone this repository: `git clone https://github.com/yourgithubusername/bitcoin-scraper.git`
2. Install PRAW: `pip install praw`
3. Create a new Reddit app and get your API credentials
4. Rename `praw.sample.ini` to `praw.ini` and enter your API credentials

## Usage

To run the scraper, run the following command:

```
python scraper.py
```

The script will extract posts from the Bitcoin subreddit and save them to a CSV file.

## Customization

You can customize the script to extract data from other subreddits by changing the `subreddit_name` variable in `scraper.py`. You can also modify the data that is being extracted by changing the `post_data` dictionary in `scraper.py`. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was inspired by [python subreddit scraper](https://github.com/bancolombia/python-subreddit-scraper).