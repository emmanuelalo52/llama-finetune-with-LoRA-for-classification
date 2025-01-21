from .Dependecies import *
#List out the data set available
class_dataset = list_datasets(full=True, filter="text-classification")
classification_dataset_names = [dataset.id for dataset in class_dataset]

#choose one of the text classification data
tweet_sent = load_dataset("ckandemir/bitcoin_tweets_sentiment_kaggle")

#cleaning the dataset
"""
Clean out the data:
                    From URL, mentions, encode the dataset, Remove empty/NaN rows, map all changes through the dataset
"""
tweet_sent.set_format(type="pandas")
df = tweet_sent["train"][:]

#Converting strings to int 
def convert_str_int_col(df,column_name,custom_map=None):
    #check where the empty sets are
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataframe")
    if df[column_name].isnull().any():
        raise ValueError(f"Column '{column_name}' contains missing values")
    if custom_map:
        unique_values = df[column_name].unique()
        for value in unique_values:
            if value not in custom_map:
                raise ValueError(f"Value '{value}' in column '{column_name}' is not found in custom map")
        df[f"{column_name}_int"] = df[column_name].map(custom_map)
        mapping = custom_map
    else:
        Label_encoder = LabelEncoder()
        df[f"{column_name}_int"] = Label_encoder.fit_transform(df[column_name])
        mapping = dict(zip(Label_encoder.classes_,Label_encoder.transform(Label_encoder.classes_)))
    return df,mapping

# custom_map = {"Positive": 1, "Neutral": 0, "Negative": 2}
# tweet_sent_df, mapping = convert_str_int_col(tweet_sent_df, "Sentiment", custom_map)

class FeatureCleaner(nn.Module):
    """
    A feature cleaning module for text preprocessing.
    Performs operations like URL removal, hashtag removal, stopword removal, etc.
    """

    def __init__(self, slang_dict=None):

        super().__init__()
        # Load stopwords
        self.stop_words = set(stopwords.words("english"))
        # Define slang dictionary (customizable)
        self.slang_dict = slang_dict or {
            "HODL": "hold on for dear life",
            "FOMO": "fear of missing out",
        }
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r"http\S+|www\S+|https\S+", re.MULTILINE)
        self.hashtag_pattern = re.compile(r"@\w+|#\w+")
        self.special_char_pattern = re.compile(r"[^a-zA-Z0-9\s]")
        self.date_pattern = re.compile(
            r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b|\b\w{3,9}\s\d{1,2},?\s\d{4}\b"
        )
        # Compile slang replacement pattern
        self.slang_pattern = re.compile("|".join(re.escape(key) for key in self.slang_dict.keys()))

    def remove_url(self, text):
        if not text:
            return text
        return self.url_pattern.sub("", text)

    def remove_hashtags(self, text):
        if not text:
            return text
        return self.hashtag_pattern.sub("", text)

    def remove_special_characters(self, text):
        if not text:
            return text
        return self.special_char_pattern.sub("", text)

    def to_lowercase(self, text):
        if not text:
            return text
        return text.lower()

    def demoji(self, text):
        if not text:
            return text
        return emoji.demojize(text)

    def remove_stop_words(self, text):
        if not text:
            return text
        return " ".join([word for word in text.split() if word not in self.stop_words])

    def expand_slangs(self, text):
        if not text:
            return text
        return self.slang_pattern.sub(lambda x: self.slang_dict[x.group()], text)

    def remove_dates(self, text):
        if not text:
            return text
        return self.date_pattern.sub("", text)

    def forward(self, text, remove_stopwords=True):
        if not text:
            return text

        # Apply cleaning operations in sequence
        text = self.remove_url(text)
        text = self.remove_hashtags(text)
        text = self.remove_special_characters(text)
        text = self.to_lowercase(text)
        text = self.demoji(text)
        text = self.expand_slangs(text)
        text = self.remove_dates(text)

        if remove_stopwords:
            text = self.remove_stop_words(text)

        return text
    
#run it as
#tweet_sent_df['text'] = tweet_sent_df['text'].apply(lambda x: feature_cleaner.forward(x))

# Count the number of empty or NaN rows in the 'text' column
# empty_row_count = tweet_sent_df['text'].isna().sum() + (tweet_sent_df['text'].str.strip() == "").sum()
# print(f"Number of empty rows: {empty_row_count}")

# # Drop empty rows
# tweet_sent_df = tweet_sent_df.dropna(subset=["text", "Sentiment", "Sentiment_int"])
# tweet_sent_df = tweet_sent_df[~(tweet_sent_df["text"].str.strip() == "")]
# tweet_sent_df['text'].isna().sum()

# Prepare test and validation datasets
# df_test = tweet_sent["test"][:]
# df_val = tweet_sent["eval"][:]

# tweet_sent_test = df_test.drop(columns=['Date'])
# tweet_sent_val = df_val.drop(columns=['Date'])

# tweet_sent_test, _ = convert_str_int_col(tweet_sent_test, "Sentiment", custom_map)
# tweet_sent_val, _ = convert_str_int_col(tweet_sent_val, "Sentiment", custom_map)