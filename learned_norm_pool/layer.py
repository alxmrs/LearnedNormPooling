import numpy
import theano
import theano.tensor as T
import lasagne as L

from lasagne.utils import as_tuple


class LearnedNorm2DLayer(L.layers.Pool2DLayer):
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0), ignore_border=True, mode='learned_norm', P=2.0,
                 **kwargs):
        super(LearnedNorm2DLayer, self).__init__(incoming, pool_size, stride=None, pad=(0, 0), ignore_border=True,
                                                 **kwargs)

        self.pool_size = as_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)

        self.pad = as_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode

        self.P = self.add_param(numpy.asarray(P), numpy.shape(numpy.asarray(P)), name='P', trainable=True)

    def get_output_for(self, input, **kwargs):
        x = input
        # z, = out
        # z_shape = self.get_outshape(x.shape, self.pool_size, self.ignore_border, self.stride, self.pad)
        z_shape = self.get_output_shape_for(x.shape)
        # if (z[0] is None) or (z[0].shape != z_shape):
        z = T.zeros(z_shape, dtype=x.dtype)

        zz = z[0]
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.pool_size
        st0, st1 = self.stride
        pad_h = self.pad[0]
        pad_w = self.pad[1]

        self.pr = pr
        self.pc = pc

        self.x_m2d = x.shape[-2]
        self.x_m1d = x.shape[-1]

        img_rows = self.x_m2d + 2 * pad_h
        img_cols = self.x_m1d + 2 * pad_w

        self.img_rows = img_rows
        self.img_cols = img_cols

        # pad the image
        if self.pad != (0, 0):
            y = numpy.zeros(
                    (x.shape[0], x.shape[1], img_rows, img_cols),
                    dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x

        self.y = y

        self.x0 = x.shape[0]
        self.x1 = x.shape[1]

        results, updates = theano.scan(fn=self.lp_norm,
                                       sequences=[T.arange(x.shape[0]),
                                                  T.arange(x.shape[1]),
                                                  T.arange(pr),
                                                  T.arange(pc)],
                                       non_sequences=[z],  # non sequence variables to pass in
                                       name='LPnormScan'
                                       )

        return results[-1]

    def lp_norm(self, n, k, r, c, z):
        '''
        Lp = ( 1/n * sum(|x_i|^p, 1..n))^(1/p) where p = 1 + ln(1+e^P)
        :param n:
        :param k:
        :param r:
        :param c:
        :param z:
        :return:
        '''
        ds0, ds1 = self.pool_size
        st0, st1 = self.stride
        pad_h = self.pad[0]
        pad_w = self.pad[1]

        row_st = r * st0
        row_end = T.minimum(row_st + ds0, self.img_rows)
        row_st = T.maximum(row_st, self.pad[0])
        row_end = T.minimum(row_end, self.x_m2d + pad_h)

        col_st = c * st1
        col_end = T.minimum(col_st + ds1, self.img_cols)
        col_st = T.maximum(col_st, self.pad[1])
        col_end = T.minimum(col_end, self.x_m1d + pad_w)

        Lp = T.pow(
                T.mean(T.pow(
                        T.abs_(T.flatten(self.y[n, k, row_st:row_end, col_st:col_end], 1)),
                        1 + T.log(1 + T.exp(self.P))
                )),
                1 / (1 + T.log(1 + T.exp(self.P)))
        )

        return T.set_subtensor(z[n, k, r, c], Lp)

