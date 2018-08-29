from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf


import os
from flask import Flask, render_template, session, redirect, url_for, flash,request
from scripts.label_image import load_graph,read_tensor_from_image_file,load_labels
from flask_wtf import FlaskForm
from wtforms import (SubmitField,FileField)
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'

# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html

class InfoForm(FlaskForm):
    '''
    This general class gets a lot of form about puppies.
    Mainly a way to go through many of the WTForms Fields.
    '''
    image = FileField(u'Image File')
    submit = SubmitField('Submit')



@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = InfoForm()

    # If the form is valid on submission (we'll talk about validation next)


    if form.is_submitted():
        file = request.files['image']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        session['filename'] = filename


        file_name = UPLOAD_FOLDER + filename
        model_file = "tf_files/retrained_graph.pb"
        label_file = "tf_files/retrained_labels.txt"
        input_height = 224
        input_width = 224
        input_mean = 128
        input_std = 128
        input_layer = "input"
        output_layer = "final_result"


        graph = load_graph(model_file)
        t = read_tensor_from_image_file(file_name,
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

        #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
        #template = "{} (score={:0.5f})"
        dlabels = []
        dresults = []
        for i in top_k:
          #print(template.format(labels[i], results[i]))
          dlabels.append(labels[i])
          dresults.append(results[i])
          if i > 0:
              break

        session['dlabel'] = dlabels[0]
        session['dresult'] = int(dresults[0]*100)


        return redirect(url_for("thankyou"))

    return render_template('home.html', form=form)


@app.route('/thankyou')
def thankyou():

    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True)
