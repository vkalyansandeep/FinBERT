from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the Financial BERT model
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    result = sentiment_pipeline(text)
    return render_template('result.html', text=text, result=result)

if __name__ == '__main__':
    app.run(debug=True)
