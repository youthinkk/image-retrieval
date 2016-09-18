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

import sys
import tarfile
import os.path
import re
import numpy as np
import tensorflow as tf
import urllib
import learning_index as index

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('MODEL_DIR', '/tmp/imagenet',
                           """Path to classify_image_graph_def.pb, """
                           """imagenet_synset_to_human_label_map.txt, and """
                           """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_integer('NUM_TOP_PREDICTIONS', 5,
                            """Display this many predictions.""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


class NodeLookup(object):
    """
    Convert integer node ID's to human readable labels.
    """

    def __init__(self, label_lookup_path=None, uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(FLAGS.MODEL_DIR,
                                             'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(FLAGS.MODEL_DIR,
                                           'imagenet_synset_to_human_label_map.txt')
        print uid_lookup_path
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """
        Load a human readable English name for each softmax node.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Load mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Load mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Load the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


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

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-FLAGS.NUM_TOP_PREDICTIONS:][::-1]
        name_score = []
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            name_score.append((human_string, score, node_id))

        return name_score


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
