import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

class TextPreprocessor(TransformerMixin):
    def __init__(self, barplot=False, wordcloud=False):
        self.barplot = barplot
        self.wordcloud = wordcloud

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Normalizing Case Folding
        X = X.str.lower()
        # Removing Punctuations
        X = X.str.replace('[^\w\s]', '', regex=True)
        X = X.str.replace("\n", '', regex=True)
        # Removing Numbers
        X = X.str.replace('\d', '', regex=True)
        # Removing Stopwords
        sw = stopwords.words('english')
        X = X.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
        # Removing Rarewords / Custom Words
        delete = pd.Series(' '.join(X).split()).value_counts()[-1000:]
        X = X.apply(lambda x: " ".join(x for x in x.split() if x not in delete))

        if self.barplot:
            # Calculating Term Frequencies
            tf = X.apply(lambda x: pd.Series(x.split(" ")).value_counts()).sum(axis=0).reset_index()
            tf.columns = ["words", "tf"]
            tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
            plt.show()

        if self.wordcloud:
            # Generating Wordcloud
            combined_text = " ".join(i for i in X)
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(combined_text)
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()

        return X




text_pipeline = Pipeline([
    ('preprocess', TextPreprocessor(barplot=True, wordcloud=True))
])



processed_text = text_pipeline.fit_transform(df['text'])


