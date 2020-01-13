#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams


class RBM(object):
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        w=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
        ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        if w is None:
            initial_w = np.asarray(
                    numpy_rng.uniform(
                        low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                        high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                        size=(n_visible, n_hidden)
                    ),
                    dtype=theano.config.floatX
            )
            w = theano.shared(value=initial_w, name='w', borrow=True)

        

