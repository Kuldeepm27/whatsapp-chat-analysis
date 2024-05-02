import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from helper import perform_sentiment_analysis  # Ensure correct import

# Set page title and icon
st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="📱")

# Define custom CSS styles
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        color: #333;
    }
    .css-13lsqb {
        font-size: 18px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Filter out unnecessary group notifications and security code messages
    filtered_df = df[~df['message'].str.contains('security code|changed the group|created group', case=False)]

    # Fetch unique users from the filtered DataFrame
    user_list = filtered_df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Filter dataframe based on selected user
        if selected_user == "Overall":
            selected_df = filtered_df
        else:
            selected_df = filtered_df[filtered_df['user'] == selected_user]

        # Calculate statistics based on selected user
        num_messages, total_words, num_media_messages, num_links = helper.fetch_stats(selected_user, selected_df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(total_words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Display analysis plots regardless of selected user
        if selected_user == "Overall" or not selected_df.empty:
            # Monthly timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, selected_df)
            if not timeline.empty:
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            else:
                st.write("No data available for monthly timeline.")

            # Daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, selected_df)
            if not daily_timeline.empty:
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            else:
                st.write("No data available for daily timeline.")

            # Activity map
            st.title('Activity Map')
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user, selected_df)
                if not busy_day.empty:
                    fig, ax = plt.subplots()
                    ax.bar(busy_day.index, busy_day.values, color='purple')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                else:
                    st.write("No data available for most busy day.")

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, selected_df)
                if not busy_month.empty:
                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='orange')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                else:
                    st.write("No data available for most busy month.")

            # Weekly activity map
            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, selected_df)
            if not user_heatmap.empty:
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            else:
                st.write("No data available for weekly activity map.")

            # Most Busy Users
            st.title("Most Busy Users")
            x, new_df = helper.most_busy_users(filtered_df)
            if not new_df.empty:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
                st.dataframe(new_df)
            else:
                st.write("No data available for most busy users.")

            # Display word cloud
            st.title("Word Cloud")
            df_wc = helper.create_wordcloud(selected_user, filtered_df)
            if df_wc is not None:
                st.image(df_wc.to_array(), caption='Word Cloud')
            else:
                st.write("No data available to generate word cloud.")

            # Display most common words
            most_common_df = helper.most_common_words(selected_user, filtered_df)
            if not most_common_df.empty:
                fig, ax = plt.subplots()
                ax.barh(most_common_df['Word'], most_common_df['Count'])
                plt.xticks(rotation='vertical')
                st.title('Most Common Words')
                st.pyplot(fig)
            else:
                st.write("No data available for most common words.")

            # Sentiment Analysis
            sentiment_df = perform_sentiment_analysis(selected_df['message'])
            if not sentiment_df.empty:
                st.title("Sentiment Analysis")
                st.subheader("Sentiment Analysis Results")
                st.dataframe(sentiment_df)

                # Display sentiment analysis line chart
                st.line_chart(sentiment_df['Sentiment Score'])  # Adjusted to directly plot Sentiment Score

            else:
                st.write("No data available for sentiment analysis.")

            # Perform emoji analysis
            emoji_df = helper.emoji_helper(selected_user, df)
            st.title("Emoji Analysis")

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df['Count'], labels=emoji_df['Emoji'].head(), autopct="0.2f")
                st.pyplot(fig)




