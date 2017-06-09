"""9.2 多GPU并行"""

import os.path
import re
import time
import numpy as np
import tensorflow as tf
import ch9.cifar10 as cifar10

batch_size = 128
max_steps = 1000000
num_gpus = 2

def tower_loss(scope):
    images, labels = cifar10.distorted_inputs()
    logits = cifar10.inference(images)
    _ = cifar10.loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss