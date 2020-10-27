"""
Author: Gabriele Valvano
Adapted from: https://github.com/MiguelMonteiro/CRFasRNNLayer
"""
"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana , Miguel Monteiro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import tensorflow as tf
from idas.tf_utils import get_shape

dir_path = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.abspath(os.curdir)


def run_once(f):
    """ Decorator to run a function only once. """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
        # else:
        #     pass
    wrapper.has_run = False
    return wrapper


def reset():
    """ Reset of CRF as RNN layer. """
    # TODO: automatically run reset when compiling error
    #  The problem: when we have a mismatch between actual compiled filters and what is written in config.txt
    #  you cannot automatically compile again until you delete the config file
    file_name = os.path.join(dir_path, "permutohedral_lattice/config.txt")
    os.remove(file_name)
    print('\nReset: done.\n')


@run_once
def write_config_file(config):
    file_name = os.path.join(dir_path, "permutohedral_lattice/config.txt")
    with open(file_name, "w") as file:
        for key, value in zip(config.keys(), config.values()):
            file.write("{0}={1}\n".format(key, value))


@run_once
def not_built_yet(config):
    file_name = os.path.join(dir_path, "permutohedral_lattice/config.txt")
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            for line in file:
                key, value = line.split("=")
                if str(config[key]) != value.rsplit('\n')[0]:
                    return False
    return True


@run_once
def build_kernels(std_out=None):
    if std_out is None:
        std_out = '/dev/null'
    print(' | Building the layer...')
    command = 'cd {0} && sh build.sh > {1} && cd {2}'.format(dir_path, std_out, project_dir)
    os.system(command)
    print(' | Done.')


def crf_rnn_layer(unaries, reference_image, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    # ------------------------------------------------------------------------------------
    # Build the kernels for the layer (only for first time the function is called)
    shape = unaries.shape.as_list()
    config = {
        "SPATIAL_DIMS": len(shape[1:-1]),
        "INPUT_CHANNELS": shape[-1],
        "REFERENCE_CHANNELS": reference_image.shape.as_list()[-1],
        "MAKE_TESTS": False
    }
    if not_built_yet(config):
        print(' | Built kernel not found:')
        write_config_file(config)
        build_kernels()
    else:
        print(' | Skipping build of CRF-as-RNN layer because a config file with the same specs already exists.')

    from architectures.layers.crf_as_rnn import lattice_filter_op_loader
    custom_module = lattice_filter_op_loader.module

    # ------------------------------------------------------------------------------------
    # define layer:

    with tf.variable_scope('crf_as_rnn_layer'):
        spatial_ker_weights = tf.get_variable('spatial_ker_weights', shape=(num_classes),
                                              initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1))
        bilateral_ker_weights = tf.get_variable('bilateral_ker_weights', shape=(num_classes),
                                                initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1))
        spatial_ker_weights = tf.diag(spatial_ker_weights)
        bilateral_ker_weights = tf.diag(bilateral_ker_weights)
        compatibility_matrix = tf.get_variable('compatibility_matrix', shape=(num_classes, num_classes),
                                               initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1))

        # Prepare filter normalization coefficients
        unaries_shape = get_shape(unaries)
        # all_ones = np.ones(unaries_shape, dtype=np.float32)

        q_values = unaries
        for i in range(num_iterations):

            q_values = tf.nn.softmax(q_values)

            # Spatial filtering
            # spatial_out = custom_module.high_dim_filter(q_values, reference_image, bilateral=False, theta_gamma=theta_gamma)
            spatial_out = custom_module.lattice_filter(q_values, reference_image, bilateral=False, theta_gamma=theta_gamma)

            # Bilateral filtering
            # bilateral_out = custom_module.high_dim_filter(q_values, reference_image, bilateral=True, theta_alpha=theta_alpha, theta_beta=theta_beta)
            bilateral_out = custom_module.lattice_filter(q_values, reference_image, bilateral=True, theta_alpha=theta_alpha, theta_beta=theta_beta)

            # Weighting filter outputs
            message_passing = tf.matmul(spatial_ker_weights,
                                        tf.transpose(tf.reshape(spatial_out, (-1, num_classes)))) + \
                              tf.matmul(bilateral_ker_weights,
                                        tf.transpose(tf.reshape(bilateral_out, (-1, num_classes))))

            # Compatibility transform
            pairwise = tf.matmul(compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(tf.transpose(pairwise), unaries_shape)
            q_values = unaries - pairwise

        return q_values
