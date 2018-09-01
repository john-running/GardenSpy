'''
imports and config for image recognition
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image   # for image resizing


UPLOAD_FOLDER = '/home/sidearmjohnny/mysite/static/' #remote
#UPLOAD_FOLDER = 'static/' #local
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

'''
imports for web app
'''

import os #required to save uploaded files
from flask import Flask, render_template, session, redirect, url_for, flash,request
from scripts.label_image import load_graph,read_tensor_from_image_file,load_labels,ImageFileRequired
from flask_wtf import FlaskForm
from wtforms import (SubmitField,FileField, Form, BooleanField, StringField, validators)
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'mysecretkey'
origi

class ImageForm(FlaskForm):
    '''
    This general class gets a lot of form about puppies.
    Mainly a way to go through many of the WTForms Fields.
    '''
    image = FileField('Image File', validators=[DataRequired(), ImageFileRequired()])

    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = ImageForm()

    if form.validate_on_submit(): # if user submits form
        file = request.files['image']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # save original file to the server
        session['filename'] = filename # store file name in session
        file_location = UPLOAD_FOLDER + filename

        '''
        resize image to prepare it for image recognition (expects images that are 224px wide)
        '''
        basewidth = 224
        img = Image.open(file_location)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        img.save(file_location)

        '''
        image classification from tensorflow for poets code lab example
        '''
        model_file = "/home/sidearmjohnny/mysite/tf_files/retrained_graph.pb"  #remote
        label_file = "/home/sidearmjohnny/mysite/tf_files/retrained_labels.txt" #remote
#        model_file = "tf_files/retrained_graph.pb" #local
#        label_file = "tf_files/retrained_labels.txt" #local
        input_height = 224
        input_width = 224
        input_mean = 128
        input_std = 128
        input_layer = "input"
        output_layer = "final_result"


        graph = load_graph(model_file)
        t = read_tensor_from_image_file(file_location,
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);

        with tf.Session(graph=graph) as sess:
          start = time.time()
          results = sess.run(output_operation.outputs[0],
                            {input_operation.outputs[0]: t})
          end=time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

        dlabels = [] # store results of classification in an array
        dresults = []
        for i in top_k:
          dlabels.append(labels[i])
          dresults.append(results[i])


        session['dlabel'] = dlabels[0]  #store top result in session
        session['dresult'] = int(dresults[0]*100) #store top result certainty afer convertin to percentage

        return redirect(url_for("index"))

    return render_template('home.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
