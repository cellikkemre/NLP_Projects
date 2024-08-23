import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from textblob import Word
from nltk.sentiment import SentimentIntensityAnalyzer

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    This class performs text preprocessing:
    - Converts text to lowercase
    - Removes punctuation
    - Removes numbers
    - Removes stopwords
    - Performs lemmatization
    """

    def __init__(self):
        self.stopwords = stopwords.words('english')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.str.lower()  # Convert text to lowercase
        X = X.str.replace('[^\w\s]', '', regex=True)  # Remove punctuation
        X = X.str.replace('\d', '', regex=True)  # Remove numbers
        X = X.apply(lambda x: " ".join(x for x in x.split() if x not in self.stopwords))  # Remove stopwords
        delete = pd.Series(' '.join(X).split()).value_counts()[-1000:]  # Identify rare words
        X = X.apply(lambda x: " ".join(x for x in x.split() if x not in delete))  # Remove rare words
        X = X.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))  # Perform lemmatization
        return X

def sentiment_labeling(dataframe):
    """
    Labels reviews as positive, negative, or neutral using SentimentIntensityAnalyzer.
    """
    sia = SentimentIntensityAnalyzer()

    # Ensure all reviews are string type and remove NaN values
    dataframe['Review'] = dataframe['Review'].astype(str)
    dataframe = dataframe[dataframe['Review'].notnull()]

    # Apply sentiment labeling
    dataframe["Sentiment_Label"] = dataframe["Review"].apply(lambda x: 'pos' if sia.polarity_scores(x)["compound"] > 0 else 'neg')
    return dataframe

def main():
    # Load the data
    df = pd.read_excel("KOZMOS_Sentiment_Analysis/amazon.xlsx")

    # Perform sentiment labeling
    df = sentiment_labeling(df)

    # Create the pipeline
    pipeline = Pipeline([
        ('text_preprocessor', TextPreprocessor()),  # Text preprocessing
        ('tfidf', TfidfVectorizer()),  # TF-IDF vectorization
        ('model', RandomForestClassifier(random_state=42))  # Random Forest model
    ])

    # Split the data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(df["Review"], df["Sentiment_Label"], random_state=42)

    # Train the model and evaluate performance without parallel processing
    accuracy = cross_val_score(pipeline, test_x, test_y, cv=5, n_jobs=1).mean()
    print(f'Random Forest Model Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()
