import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess the text by removing stopwords and non-alphanumeric characters
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Load the dataset (Make sure the file path is correct and properly escaped)
dataset = pd.read_csv(r'C:\abbu project\t-hub\git-chatbot\conversation_dataset.csv')

# Clean the questions and answers
dataset['cleaned_question'] = dataset['question'].apply(preprocess_text)
dataset['cleaned_answer'] = dataset['answer'].apply(preprocess_text)

# Separate features (questions) and target (answers)
X = dataset['cleaned_question']
y = dataset['cleaned_answer']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train a logistic regression classifier
classifier = LogisticRegression(max_iter=200)  # Set a higher max_iter if needed
classifier.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer using pickle
with open('chatbot_model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('cyber.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Optional: Test model accuracy
X_test_tfidf = vectorizer.transform(X_test)
accuracy = classifier.score(X_test_tfidf, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
