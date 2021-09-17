from flask import render_template, Flask, request, url_for
from keras.models import load_model
import pickle
import tensorflow.compat.v1 as tf
from gevent.pywsgi import WSGIServer
import os



app = Flask(__name__)
graph = tf.get_default_graph()

with open(r'count_vec.pkl','rb') as file:
    cv = pickle.load(file)

@app.route('/')
def home():
    return render_template('gproject.html')

@app.route('/tpredict')
@app.route('/', methods = ['GET','POST'])
def page2():
    if request.method =='GET':
        return render_template('gproject.html')
    if request.method == 'POST':
        topic = request.form['tweet']
        print(topic)
        topic = cv.transform([topic])
        print("\n"+str(topic.shape) + "\n")
        with graph.as_default():
            cla = load_model('review.h5')
            cla.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            ypred = cla.predict(topic)
            print("pred is" + str(ypred))
        if(ypred > 0.5):
            topic = "Positive review"
        else:
            topic = "Negative review"
        
        return render_template('gproject.html', ypred = topic)

if __name__ == "__main__":
    app.run(port=8086, debug=True)
