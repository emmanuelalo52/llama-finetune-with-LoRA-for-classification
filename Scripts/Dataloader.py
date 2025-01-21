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

