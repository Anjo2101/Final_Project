# Final Project (Project 4)
Dataset can be accessed using this URL: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data

Here is my write up for the project:

# 🎬 IMDb Movie Review Sentiment Analysis

This project applies machine learning techniques to classify IMDb movie reviews as either **positive** or **negative** based on review text. It uses NLP preprocessing, vectorization, and multiple classification models to explore performance.

---

## 📁 Dataset

- **Source**: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
- **Size**: 50,000 reviews (balanced: 25,000 positive, 25,000 negative)

---

## 🧰 Technologies Used

- **Languages & Libraries**: Python, Pandas, Seaborn, Matplotlib, BeautifulSoup, Regex
- **Machine Learning & NLP**: `scikit-learn` (CountVectorizer, TfidfVectorizer, LogisticRegression, RandomForestClassifier)

---

## 🧼 Preprocessing Steps

- Removal of HTML tags using `BeautifulSoup`
- Unicode and ASCII normalization
- Elimination of URLs, square brackets, and special characters
- Conversion to lowercase for text normalization

---


✅ Balanced dataset ideal for classification.

---

## 🔧 Feature Engineering

Two vectorization techniques were applied:

- **Bag-of-Words** (CountVectorizer)
- **TF-IDF** (TfidfVectorizer)

---

## 🤖 Models Trained

| Model                  | Vectorizer     | Accuracy |
|------------------------|----------------|----------|
| Logistic Regression    | CountVectorizer | 89.3%    |
| Logistic Regression    | TF-IDF          | 89.6%    |
| Random Forest Classifier | TF-IDF        | 84.1%    |

### 📈 Classification Report Highlights

- **Logistic Regression + TF-IDF** had the best precision and recall (≈ 90%)
- **Random Forest** slightly underperformed due to high-dimensional sparse data

---

## ⚠️ Challenges Encountered

- Handling malformed or encoded text in preprocessing
- Convergence issues with Logistic Regression (`max_iter` tuning needed)
- Shape mismatches during label binarization

---

## ✅ Conclusion

- TF-IDF provided better results than Bag-of-Words
- Logistic Regression proved to be highly effective for this binary classification task
- Simpler models, when well-preprocessed, can deliver excellent performance

---

## 🚀 Future Work

- Try deep learning models like LSTM or transformers (BERT)
- Add grid search & cross-validation for hyperparameter tuning
- Explore stacking or ensemble methods

---

## 📌 Results Summary

```python
# Accuracy Scores
Bag of Words + Logistic Regression:  89.3%
TF-IDF + Logistic Regression:        89.6%
TF-IDF + Random Forest:              84.1%


