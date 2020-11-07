# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Support Vector Machine model and TFIDF object from disk
filename = 'suicidal-svc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
tfidf = pickle.load(open('tfidf-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = tfidf.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)