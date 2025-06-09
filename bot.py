import pandas as pd
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Suppress downloader messages
logging.getLogger('nltk.downloader').setLevel(logging.ERROR)

# Download NLTK stopwords
#nltk.download('stopwords')

# Setup
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv('sample_data.csv')  # Make sure it has 'question' and 'answer' columns
data.dropna(subset=['question', 'answer'], inplace=True)

# Text cleaning
def clean_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Spell correction only
def autocorrect_text(text):
    return str(TextBlob(text).correct())

# Preprocess questions
data['clean_question'] = data['question'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(data['clean_question'])

# Chatbot logic
def get_response(user_input):
    corrected_input = autocorrect_text(user_input)
    cleaned_input = clean_text(corrected_input)
    user_vec = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    confidence = similarity[0][idx]
    if confidence < 0.2:
        return "🤖 Sorry, I don't understand that."
    return data['answer'].iloc[idx]

# CLI Chatbot
print("🤖 Welcome to ChatBot! Type 'exit' to quit.")
while True:
    user = input("You: ")
    if user.lower() == 'exit':
        print("Bot: Goodbye! 👋")
        break
    response = get_response(user)
    print("Bot:", response)
