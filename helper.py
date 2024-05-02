from collections import Counter
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
from emoji import UNICODE_EMOJI
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
from textblob import TextBlob
import re


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Number of messages
    num_messages = df.shape[0]

    # Total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # Number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # Number of links shared
    extract = URLExtract()
    num_links = df['message'].apply(lambda x: len(extract.find_urls(str(x)))).sum()

    return num_messages, len(words), num_media_messages, num_links


def most_busy_users(df):
    user_counts = df['user'].value_counts().head()
    message_percentages = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return user_counts, message_percentages


def most_common_words(selected_user, df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Combine all messages into a single string
    text = ' '.join(df['message'])

    # Tokenize the text into words
    words = text.split()

    # Count the occurrences of each word
    word_counts = Counter(words)

    # Convert the word counts to a DataFrame
    word_counts_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])

    # Sort the DataFrame by count in descending order
    word_counts_df = word_counts_df.sort_values(by='Count', ascending=False)

    return word_counts_df.head(10)


def create_wordcloud(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    if df.empty:
        raise ValueError("No messages for the selected user.")

    # Concatenate all messages into a single string
    text = " ".join(df['message'])

    # Generate the word cloud
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    return wc


def perform_sentiment_analysis(messages):
    sentiment_scores = []
    sentiment_labels = []

    for message in messages:
        try:
            # Perform sentiment analysis using TextBlob
            blob = TextBlob(str(message))
            sentiment_score = blob.sentiment.polarity

            # Determine sentiment label based on the sentiment score
            if sentiment_score > 0:
                sentiment = 'Positive'
            elif sentiment_score < 0:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            sentiment_scores.append(sentiment_score)
            sentiment_labels.append(sentiment)
        except Exception as e:
            # Handle any errors that may occur during sentiment analysis
            print(f"Error occurred during sentiment analysis: {e}")
            sentiment_scores.append(None)
            sentiment_labels.append(None)

    # Create a DataFrame to store sentiment analysis results
    analysis_df = pd.DataFrame({
        'Message': messages,
        'Sentiment Score': sentiment_scores,
        'Sentiment': sentiment_labels
    })

    return analysis_df


def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Combine all messages into a single string
    all_messages = ' '.join(df['message'])

    # Find all emojis in the combined messages
    emojis = [c for c in all_messages if c in emoji.UNICODE_EMOJI]
    print("Emojis found:", emojis)

    # Count occurrences of each emoji
    emoji_counts = Counter(emojis)
    print("Emoji counts:", emoji_counts)

    if emoji_counts:
        emoji_df = pd.DataFrame(emoji_counts.items(), columns=['Emoji', 'Count'])
    else:
        emoji_df = pd.DataFrame(columns=['Emoji', 'Count'])

    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap
