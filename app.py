# imports
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

app = Flask('myApp')

@app.route('/')
def home():
    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)