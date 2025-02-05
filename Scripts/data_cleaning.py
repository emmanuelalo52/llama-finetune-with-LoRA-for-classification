from Dependecies import *
#IF APPLICABLE TO YOUR DATASET
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