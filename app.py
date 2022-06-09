from flask import Flask, request, render_template
import pickle
from sklearn import *


# unplickling the model

file = open('campusplacementpredictor.pkl', 'rb')
rf = pickle.load(file)
file.close()


app = Flask(__name__)

# index page
@app.route('/')
def home():
    return render_template('index.html')

# courses page
@app.route('/courses')
def courses():
     return render_template('course.html')

# campus prediction
@app.route('/campus_predict')
def campus_predict():
     return render_template('campus_predict.html')
    
# back btn of course page
@app.route('/index')
def index():
     return render_template('index.html')

# redirect to html course page
@app.route('/html')
def html():
     return render_template('html.html')

# redirect to css course page
@app.route('/css')
def css():
     return render_template('css.html')

# redirect to python course page
@app.route('/python')
def python():
     return render_template('python.html')

# redirect to ml course page
@app.route('/ml')
def ml():
     return render_template('ml.html')

# redirect to java course page
@app.route('/java')
def java():
     return render_template('java.html')

# redirect to courses page
@app.route('/course')
def course():
     return render_template('course.html')

# book search page
@app.route('/book_search')
def book_search():
     return render_template('book_search.html')

# book page
@app.route('/book')
def book():
    return render_template('book.html')

# Cmapus Predict.html back button of show page

@app.route('/campus_predict_show')
def campus_predict_show():
    return render_template('campus_predict_show.html') 


@app.route('/predict', methods=['GET', 'POST'])  

def hello_world():

    if request.method == 'POST':

        mydict = request.form
        gender = int(mydict['gender'])
        spec = int(mydict['specialisation'])
        tech = int(mydict['degree_t'])
        work = int(mydict['workex'])
        ssc = float(mydict['ssc_P'])
        hsc = float(mydict['hsc_p'])
        dsc = float(mydict['degree_p'])
        mba = float(mydict['mba_p'])
        inputfeatures = [[gender, spec, tech, work, ssc, hsc, dsc, mba]]

        # predicting the class either 0 or 1

        predictedclass = rf.predict(inputfeatures)

        # predicting the probability

        predictedprob = rf.predict_proba(inputfeatures)

        print(predictedclass, predictedprob[0][0])

        if predictedclass[0] == 1:
            proba = predictedprob[0][1]

        else:
            proba = predictedprob[0][0]

        print(predictedclass, proba*100)

        placemap = {1: 'Will be Placed', 0: 'Better Luck Next Time :('}
        predictedclasssend = placemap[predictedclass[0]]

        if predictedclass[0] == 1:
            return render_template('show.html', predictedclasssend=predictedclasssend, predictedprob=round(proba*100, 2), placed=True)

        else:
            return render_template('show.html', predictedclasssend=predictedclasssend)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
