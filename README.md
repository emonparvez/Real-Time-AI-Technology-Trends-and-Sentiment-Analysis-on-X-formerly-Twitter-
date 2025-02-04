# Real-Time AI Technology Trends and Sentiment Analysis on X (formerly Twitter)

## Overview

This project performs real-time sentiment analysis on AI-related tweets, specifically focusing on reactions to the launch of DeepSeek's open-source initiative. The project fetches tweet data using X (formerly Twitter) API, processes it, applies feature extraction techniques, and evaluates several machine learning models for sentiment classification. The models track sentiment trends and offer insights into public opinions related to AI technology.

## Table of Contents

- [Project Description](#overview)
- [Dependencies](#dependencies)
- [Data Collection](#data-collection)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Extraction](#feature-extraction)
- [Data Preprocessing and Handling Imbalance](#data-preprocessing-and-handling-imbalance)
- [Model Comparison](#model-comparison)
- [Real-Time Sentiment Tracking](#real-time-sentiment-tracking)
- [Results and Reporting](#results-and-reporting)
- [License](#license)

## Dependencies

To run this project, you need the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `imblearn`
- `nltk`
- `tweepy`
- `wordcloud`
- `textblob`

You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imblearn nltk tweepy wordcloud textblob

##Data Collection
Fetch Tweets
We used the X (formerly Twitter) API to fetch tweet data by querying specific keywords like "AI technology", "DeepSeek", and "open-source". The data includes tweet metadata such as the tweet’s text, date of creation, tweet length, and sentiment.

##Access the Twitter API
You will need to create a Twitter Developer account and generate API keys. The API is used to fetch real-time tweet data based on your chosen keyword query.

`import tweepy

# Your credentials here
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Query tweets
tweets = api.search_tweets("AI technology DeepSeek open-source", lang='en', count=100)`

