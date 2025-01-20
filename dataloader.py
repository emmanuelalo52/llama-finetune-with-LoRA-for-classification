from huggingface_hub import list_datasets
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes
#List out the data set available
def list_data(word:str):
    class_dataset = list_datasets(full=True, filter=word)
    classification_dataset_names = [dataset.id for dataset in class_dataset]

    print(f"There are {len(classification_dataset_names)} classification datasets available on the hub")
    print(f"The first 10 are: {classification_dataset_names[:10]}")

# list_data("text-classification")
#load the ckandemir/bitcoin_tweets_sentiment_kaggle dataset for our model
tweet_semt = load_dataset("ckandemir/bitcoin_tweets_sentiment_kaggle")
# print(tweet_semt)

train_ds = tweet_semt["train"]
# print(train_ds)

# print(train_ds[0])

#review column names
