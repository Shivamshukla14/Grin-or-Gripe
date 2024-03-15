from flask import Flask, render_template, request, Blueprint
from flask_bootstrap import Bootstrap # Import for app3

#NLP Packages for app3
from textblob import TextBlob, Word
import random
import time

import numpy as np #I Import for app1
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification #Import for app2
from scipy.special import softmax


app = Flask(__name__)
Bootstrap(app)  # Initialize Bootstrap for app3


#Blueprint for each app
app1_bp = Blueprint('app1', __name__)
app2_bp = Blueprint('app2', __name__)
app3_bp = Blueprint('app3', __name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')  # Render the about.html template

@app.route('/services/')
def services():
    return render_template('service.html')  # Render the about.html template


# --- Code for app1 (Sentiment Analysis using Pickle) ---
loaded_vectorizer = None
loaded_model = None
stemmer = PorterStemmer()

# Function for text pre-processing 
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

with open("model/model_and_vectorizer.pkl", "rb") as f:
    data = pickle.load(f)
    loaded_vectorizer = data["vectorizer"]
    loaded_model = data["model"]


@app1_bp.route('/services/app1/')
def app1_home():
    return render_template('app1.html')

@app1_bp.route('/services/app1/predict', methods = ['POST'])
def app1_predict():
    user_input = request.form["tweet"]
    processed_input = stemming(user_input)
    X_new = np.array([processed_input]).reshape(1, -1)
    processed_text = ' '.join(X_new[0])
    X_new_transformed = loaded_vectorizer.transform([processed_text])
    prediction = loaded_model.predict(X_new_transformed)
    sentiment = "Negative" if prediction[0] == 0 else "Positive"
    return render_template("result.html", sentiment = sentiment)


# --- code for app2 (Sentiment Analysis using Transformers) ---
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
model = None
tokenizer = None

try:
    model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
    tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    app.config['model_error'] = True #Flag for error handling in templates


def preprocess_tweet(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'  # Replace mentions with '@user'
        elif word.startswith('http'):
            word = 'http'  # Replace URLs with 'http'
        tweet_words.append(word)
    tweet_proc = ' '.join(tweet_words)
    return tweet_proc


@app2_bp.route('/services/app2/', methods = ["GET", "POST"])
def app2_analyze_sentiment():
    if request.method == "GET":
        return render_template("app2.html")
    else:
        tweet = request.form["tweet"]
        tweet_proc = preprocess_tweet(tweet)
        if not app.config.get('model_error'):
            try:
                encoded_tweet = tokenizer(tweet_proc, return_tensors = 'pt')
                output = model(**encoded_tweet)
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)
                labels = ['Negative', 'Neutral', 'Positive']
                sentiment_results = [(label, f"{score:.2f}%") for label, score in zip(labels, scores * 100)]
                return render_template("result.html", sentiment_results=sentiment_results)
            except Exception as e:
                print(f"Error during sentiment analysis: {e}")
                return render_template("error.html")
        else:
            return render_template("error.html")
        

@app3_bp.route('/services/app3/')
def app3_index():
    return render_template('app3.html')

@app3_bp.route('/services/app3/analyse', methods = ['POST'])
def app3_analyse():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        #NLP Stuff
        blob = TextBlob(rawtext)
        received_text2 = blob
        blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
        number_of_tokens = len(list(blob.words))
        #Extracting Main Points
        nouns = list()
        for word, tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())
                len_of_words = len(nouns)
                rand_words = random.sample(nouns, len(nouns))
                final_word = list()
                for item in rand_words:
                    word = Word(item).pluralize()
                    final_word.append(word)
                    summary = final_word
                    end = time.time()
                    final_time = end-start


    return render_template('app3.html', received_text = received_text2, number_of_tokens = number_of_tokens,
                               blob_sentiment = blob_sentiment, blob_subjectivity = blob_subjectivity, summary = summary,
                               final_time = final_time)
    

#Register Blueprints with the main app
app.register_blueprint(app1_bp)
app.register_blueprint(app2_bp)
app.register_blueprint(app3_bp)

if __name__ == '__main__':
	app.run(debug=True)