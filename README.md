# Fake Review Detection Using Linear Regression & Gradio

This project predicts product review ratings and classifies them as **Real** or **Fake** using a simple machine learning model trained on text data. It uses **NLP techniques**, **TF-IDF vectorization**, and a **Linear Regression model**, all wrapped in a clean **Gradio web interface**.

---

## Why Detect Fake Reviews?

In today's online shopping ecosystem, **fake product reviews** can mislead consumers into buying poor-quality items or avoiding good ones. Detecting them is critical to:

- 🛡️ Improve customer trust and experience
- 📉 Reduce brand reputation damage
- 💰 Save buyers from poor purchase decisions

---

## Tech Stack

| Layer                   | Tool / Library                           | Purpose                                                                 |
|-------------------------|------------------------------------------|-------------------------------------------------------------------------|
| **Data Handling**       | `pandas`                                 | Load and preprocess the review dataset from CSV format.                |
| **NLP Preprocessing**   | `nltk`, `string`                         | Clean and prepare text (remove stopwords, lowercase, remove punctuation). |
| **Text Vectorization**  | `TfidfVectorizer` (`scikit-learn`)       | Convert review text into numerical vectors using TF-IDF representation.|
| **Modeling**            | `LinearRegression` (`scikit-learn`)      | Predict numerical ratings (1–5 stars) from the review text.            |
| **Data Splitting**      | `train_test_split` (`sklearn.model_selection`) | Split data into training and testing subsets.                          |
| **Model Persistence**   | `joblib`                                 | Save and reload the trained model and vectorizer.                      |
| **Web Interface**       | `gradio`                                 | Create an interactive UI for users to test reviews and see predictions.|
| **Platform**            | Google Colab (recommended)               | Cloud-based Python environment with pre-installed libraries.           |
| **Dataset Format**      | `.csv`                                   | Source file containing product reviews and ratings.                    |
| **Output Styling**      | HTML in `gr.HTML()`                      | Show prediction result (Real or Fake) with color styling in output.    |

---

## How It Works

1. **Data Loading:** The dataset is imported from Kaggle and loaded using `pandas`.
2. **Text Cleaning:** Reviews are converted to lowercase, stripped of punctuation, and filtered to remove common stopwords using `nltk`.
3. **Vectorization:** Cleaned text is transformed into TF-IDF vectors to represent the importance of each word in a numerical format.
4. **Model Training:** A `LinearRegression` model is trained on these vectors to predict the **star rating** of the review.
5. **Classification:** If the predicted rating is **≥ 3**, the review is marked **Real** (green). Otherwise, it is marked **Fake** (red).
6. **Gradio Interface:** Users can enter a product review in a textbox and instantly see the prediction result.

---
## Conclusion

- This project demonstrates how simple NLP and machine learning techniques can be used to detect potentially fake reviews based on text sentiment.
- By combining `TF-IDF`, `Linear Regression`, and a `Gradio UI`, the app provides a quick and interactive way to test review authenticity.
- It serves as a solid base for more advanced models using classification algorithms like SVM, XGBoost, or neural networks.

---

### Dataset Source:
- [Kaggle - Fake Product Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

---
