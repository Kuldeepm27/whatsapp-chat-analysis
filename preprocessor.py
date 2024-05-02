import pandas as pd
import re


def preprocess(data):
    if not isinstance(data, str):
        raise ValueError("Input data must be a string.")

    # Define pattern to extract datetime information
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s(?:1[0-2]|0?[1-9]):[0-5][0-9]\s?[ap]m)\s-\s'
    matches = re.findall(pattern, data)

    if not matches:
        raise ValueError("No messages or dates found in the input data.")

    dates = []
    messages = []

    for match in matches:
        # Split data based on the match to separate datetime and message
        split_data = re.split(match, data, maxsplit=1)
        if len(split_data) == 2:
            dates.append(match)
            messages.append(split_data[1])

    if len(messages) != len(dates):
        raise ValueError("Mismatch between the number of messages and dates.")

    # Create DataFrame with extracted dates and messages
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []

    # Extract user and message from user_message column
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if len(entry) > 1:
            user = entry[1]
            message_text = entry[2] if len(entry) > 2 else ""
        else:
            user = 'group_notification'
            message_text = entry[0]

        # Skip specific group notifications
        if "security code" in message_text or "changed the group" in message_text:
            continue

        users.append(user)
        messages.append(message_text)

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract additional datetime features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Define message periods based on hour
    period = []

    for hour in df['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df


def clean_message(message):
    # Remove URLs, emojis, and other unwanted characters
    cleaned_message = re.sub(r'http\S+', '', message)  # Remove URLs
    cleaned_message = re.sub(r'<[^>]+>', '', cleaned_message)  # Remove HTML tags
    cleaned_message = ' '.join(re.sub("[^a-zA-Z]", " ", cleaned_message).split())  # Remove non-alphanumeric characters

    return cleaned_message


def tokenize_message(message):
    # Tokenize messages into words
    return message.split()
