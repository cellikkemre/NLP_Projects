import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word,TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x: '%.2f' % x)
pd.set_option('display.width',200)


##################################################
# 1. TEXT PRE-PROCESSING
##################################################

df = pd.read_excel("KOZMOS_Sentiment_Analysis/amazon.xlsx")
df.info()
df.head()


df.tail()



# Normalizing Case Folding
############################
df['Review'] = df['Review'].str.lower()

############################
# Punctuations
############################
df['Review'] = df['Review'].str.replace('[^\w\s]', '',regex=True)

############################
# Numbers
############################
df['Review'] = df['Review'].str.replace('\d', '',regex=True)


############################
# Stopwords
############################
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


############################
# Rarewords / Custom Words
############################
delete = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))

############################
# Lemmatization
############################

df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df['Review'].head(10)


############################
# 2.Data Visualization
############################

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()


############################
# Wordcloud
############################

text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()




########################################################
# 3.Sentiment Analysis and Feature Engineering
########################################################

df.head()

sia = SentimentIntensityAnalyzer()

# I calculate polarity_scores() for the first 10 observations of the variable "Review"
################################################################################################################
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

# I observe again by filtering according to compound scores for the first 10 observations examined.
################################################################################################################
df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# If the compound scores for 10 observations are greater than 0, I update them as "pos", otherwise as "neg".
################################################################################################################
df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")


# I assign pos-neg to all observations in the "Review" variable and add it to the dataframe as a new variable
################################################################################################################
df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df.groupby("Sentiment_Label")["Star"].mean()


########################################################
# 4.Prepare for Machine Learning!
########################################################

#Test-Train
################
train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

#TF-IDF Word Level
#####################
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)


########################################################
# 5. Modeling (Logistic Regression)
########################################################
log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()
#0.8546034570411795

# Asking the model randomly from the comments in the series
################################################################################################################
random_review = pd.Series(df["Review"].sample(1).values)
new_comment = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(new_comment)
print(f'Review:  {random_review[0]} \n Prediction: {pred}')


###############################
# 6:  Modeling  (Random Forest)
###############################
rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()
#0.8959430604982206
