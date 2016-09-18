"""
Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os.path
import sys
import tarfile
import urllib

import numpy as np
import tensorflow as tf

from src.core.learning_index import load_index

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('MODEL_DIR', '/tmp/imagenet',
                           """Path to classify_image_graph_def.pb, """
                           """imagenet_synset_to_human_label_map.txt, and """
                           """imagenet_2012_challenge_label_map_proto.pbtxt.""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


def create_graph():
    """
    Create a graph from saved GraphDef file and returns a saver.
    """
    download_model()
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(FLAGS.MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def inference(image_path):
    """
    Run inference on an image.
    """
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        return predictions


def learning_similarity(predictions, target_label):
    index = load_index(target_label)
    max_probability = 0.

    for i in xrange(len(index)):
        max_probability = max(max_probability, predictions[index[i]])

    return max_probability


def download_model():
    """
    Download and extract model tar file.
    """
    destination_directory = FLAGS.MODEL_DIR
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    file_name = DATA_URL.split('/')[-1]
    file_path = os.path.join(destination_directory, file_name)
    if not os.path.exists(file_path):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                file_name, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        file_path, _ = urllib.urlretrieve(DATA_URL, file_path, _progress)
        stat_info = os.stat(file_path)
        print "\nSuccessfully downloaded ", file_name, stat_info.st_size, "bytes."

    tarfile.open(file_path, 'r:gz').extractall(destination_directory)
