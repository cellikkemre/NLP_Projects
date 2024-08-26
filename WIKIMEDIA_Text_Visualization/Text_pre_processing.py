import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)


df = pd.read_csv("WIKIMEDIA_Text_Visualization/Data/wiki_data.csv", index_col=0)
df.head()
df = df[:2000]

df.head()
df.shape


########################################################
# Normalizing Case Folding & Punctuations & Numbers
########################################################
def clean_text(text):
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace(r'[^\w\s]', '', regex=True)
    text = text.str.replace("\n" , '', regex=True)
    # Numbers
    text = text.str.replace('\d', '', regex=True)
    return text

df["text"] = clean_text(df["text"])

df.head()


############################
# Stopwords
############################
def remove_stopwords(text):
    stop_words = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

df["text"] = remove_stopwords(df["text"])



############################
# Rarewords / Custom Words
############################
pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
delete = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))


############################
# Tokenize
############################
df["text"].apply(lambda x: TextBlob(x).words)

############################
# Lemmatization
############################

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head()

############################
# 2.Data Visualization
############################

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.head()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

############################
# Wordcloud
############################
text = " ".join(i for i in df["text"])
wordcloud = WordCloud(max_font_size=50,
max_words=100,
background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


