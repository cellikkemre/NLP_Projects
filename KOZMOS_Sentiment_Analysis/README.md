# KOZMOZ: Sentiment Analysis for Amazon Reviews

## Project Description

This project aims to analyze the reviews of **Kozmos**, a company that produces home textiles and casual wear, selling products on Amazon. The goal is to analyze customer reviews using sentiment analysis to label them as positive or negative and then use this labeled data to build classification models and compare their performance.

## Dataset

The dataset consists of variables like reviews, titles, star ratings, and the number of users who found the review helpful for a specific product category.

- 4 Variables
- 5611 Observations
- 489 KB Size

## Steps

1. **Text Preprocessing:**
   - Applied lowercasing, punctuation removal, numeric removal, stopword removal, rare word removal, and lemmatization.

2. **Text Visualization:**
   - Generated a bar plot for word frequencies.
   - Visualized the most frequently used words using WordCloud.

3. **Sentiment Analysis:**
   - Used NLTK's **SentimentIntensityAnalyzer** to calculate the sentiment score of each review and labeled them as "positive" or "negative."

4. **Preparing for Machine Learning:**
   - Split the data into train and test sets.
   - Used TF-IDF vectorization to convert the text data into numerical features.

5. **Modeling (Logistic Regression and Random Forest):**
   - Built models using Logistic Regression and Random Forest classifiers.
   - Evaluated and compared the model performances and reported the accuracy.

## Results

- Logistic Regression achieved 85.46% accuracy.
- Random Forest achieved 89.59% accuracy.
- Random Forest outperformed Logistic Regression in this analysis.

---

### Python Code to Generate README File

```python
content = """
# KOZMOZ: Sentiment Analysis for Amazon Reviews

## Project Description

This project aims to analyze the reviews of **Kozmos**, a company that produces home textiles and casual wear, selling products on Amazon. The goal is to analyze customer reviews using sentiment analysis to label them as positive or negative and then use this labeled data to build classification models and compare their performance.

## Dataset

The dataset consists of variables like reviews, titles, star ratings, and the number of users who found the review helpful for a specific product category.

- 4 Variables
- 5611 Observations
- 489 KB Size

## Steps

1. **Text Preprocessing:**
   - Applied lowercasing, punctuation removal, numeric removal, stopword removal, rare word removal, and lemmatization.

2. **Text Visualization:**
   - Generated a bar plot for word frequencies.
   - Visualized the most frequently used words using WordCloud.

3. **Sentiment Analysis:**
   - Used NLTK's **SentimentIntensityAnalyzer** to calculate the sentiment score of each review and labeled them as "positive" or "negative."

4. **Preparing for Machine Learning:**
   - Split the data into train and test sets.
   - Used TF-IDF vectorization to convert the text data into numerical features.

5. **Modeling (Logistic Regression and Random Forest):**
   - Built models using Logistic Regression and Random Forest classifiers.
   - Evaluated and compared the model performances and reported the accuracy.

## Results

- Logistic Regression achieved 85.46% accuracy.
- Random Forest achieved 89.59% accuracy.
- Random Forest outperformed Logistic Regression in this analysis.
"""

