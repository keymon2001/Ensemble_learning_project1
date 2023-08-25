# Main python code for calling object
# Libraries

from flask import Flask, render_template, request, session
import pandas as pd
import os
import ensembleml as em
from fileinput import filename
from werkzeug.utils import secure_filename

# Flask constructor
app = Flask(__name__)
img = os.path.join('static')
# Root endpoint
@app.route('/',methods = ["GET", "POST"])
def home():
	file = os.path.join(img, 'ensLogo.png')
	return render_template('index.html',)


@app.route('/view',methods = ["GET", "POST"])
def view():
	url = request.form['text']# read url
	# Read the url using Flask request
	print("Urlo-------------",url)
	#python class
	infotext = "datainformation.txt"
	ml = em.emsrmbleML(url,infotext)
	ml.variablepro()
	ml.calculationem()
	df = ml.result
	b_lines = [row for row in (list(open("datainformation.txt")))]
	return render_template('new.html',tables=[df.to_html(classes='table table-stripped')],b_lines=b_lines,titles = ['na', 'emsemble models:'])

# Main Driver Function
if __name__ == '__main__':
	# Run the application on the local development server
	app.run(debug=True)