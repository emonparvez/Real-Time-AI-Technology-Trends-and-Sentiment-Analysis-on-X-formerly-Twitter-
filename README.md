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



## Exploratory Data Analysis (EDA)

The data was cleaned and analyzed using basic statistics and visualizations. Key steps include:

- Removing duplicates and unnecessary columns.
- Handling missing values.
- Visualizing the distribution of sentiment labels and tweet lengths.
- Creating word clouds and frequency distributions.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualizing sentiment distribution
sns.countplot(data=tweet_df, x='sentiment')
plt.show()

## Feature Extraction

Text data was transformed into numerical representations using two methods:

### 1. Bag of Words (BoW)
- Converts each word in a tweet into a unique token.
- Counts the frequency of these tokens in each tweet.

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
- Measures the importance of a word in a tweet relative to all other tweets.
- Helps reduce the impact of commonly used words while emphasizing rare but significant words.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Initialize the vectorizers
vectorizer_bow = CountVectorizer(stop_words='english')
vectorizer_tfidf = TfidfVectorizer(stop_words='english')

# Apply to cleaned text data
X_bow = vectorizer_bow.fit_transform(tweet_df['cleaned_text'])
X_tfidf = vectorizer_tfidf.fit_transform(tweet_df['cleaned_text'])


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Initialize the vectorizers
vectorizer_bow = CountVectorizer(stop_words='english')
vectorizer_tfidf = TfidfVectorizer(stop_words='english')

# Apply to cleaned text data
X_bow = vectorizer_bow.fit_transform(tweet_df['cleaned_text'])
X_tfidf = vectorizer_tfidf.fit_transform(tweet_df['cleaned_text'])


## Data Preprocessing and Handling Imbalance

To handle class imbalance (e.g., an overwhelming number of positive sentiments), **SMOTE (Synthetic Minority Over-sampling Technique)** was applied. SMOTE generates synthetic samples for the minority class, helping to balance the dataset and improve model performance.

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_combined, tweet_df['sentiment'])


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Initialize the vectorizers
vectorizer_bow = CountVectorizer(stop_words='english')
vectorizer_tfidf = TfidfVectorizer(stop_words='english')

# Apply to cleaned text data
X_bow = vectorizer_bow.fit_transform(tweet_df['cleaned_text'])
X_tfidf = vectorizer_tfidf.fit_transform(tweet_df['cleaned_text'])

## Data Preprocessing and Handling Imbalance

To handle the class imbalance (e.g., many positive sentiments), **SMOTE (Synthetic Minority Over-sampling Technique)** was used. SMOTE generates synthetic samples for the minority class, balancing the dataset and improving model performance.

### Applying SMOTE

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_combined, tweet_df['sentiment'])


from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_combined, tweet_df['sentiment'])


## Model Comparison

Several machine learning models were compared to classify sentiment into **positive, negative, and neutral** categories:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **Gradient Boosting**
- **Naive Bayes**
- **XGBoost**

Each model was evaluated using key performance metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### Training and Evaluating Models

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

## Real-Time Sentiment Tracking

The final model tracks sentiment on **real-time AI-related tweets**, providing valuable insights into public opinion trends. The results are stored in an **Excel file** for further analysis and reporting.

### Saving Results to Excel

```python
import pandas as pd

# Save results to an Excel file
results = pd.DataFrame({'tweet': tweet_texts, 'sentiment': predicted_sentiments})
results.to_excel('sentiment_results.xlsx', index=False)


import pandas as pd

# Save results to an Excel file
results = pd.DataFrame({'tweet': tweet_texts, 'sentiment': predicted_sentiments})
results.to_excel('sentiment_results.xlsx')


###Results and Reporting
The model was successful in classifying tweets into positive, negative, and neutral categories. The analysis provided insights into public sentiment surrounding AI technology and the DeepSeek open-source initiative.

###Example Output:
Positive sentiment dominated discussions.
Real-time sentiment analysis shows how opinions evolved after DeepSeek's launch.















