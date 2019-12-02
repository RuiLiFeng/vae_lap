from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class AbstractArch(object):
    def __init__(self, name, exceptions=None):
        self.name = name
        assert isinstance(exceptions, (list, tuple)) or exceptions is None
        self.exceptions = exceptions

    @property
    def trainable_variables(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

    @property
    def restore_variables(self):
        var_list = [var for var in tf.global_variables() if self.name in var.name]
        if self.exceptions is not None:
            for exception in self.exceptions:
                var_list = [var for var in var_list if exception not in var.name]
        return var_list


