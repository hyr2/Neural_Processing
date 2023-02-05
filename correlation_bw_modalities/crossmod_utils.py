import os

import numpy as np
from scipy.io import loadmat
import scipy.signal as signal


# FROM https://github.com/scipy/scipy/blob/v1.10.0/scipy/signal/_signaltools.py#L290-L384
def my_correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.
    Parameters
    ----------
    in1_len : int
        First input size.
    in2_len : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.
    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.
    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt
    Where :math:`\tau` is defined as the displacement, also known as the lag.
    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]
    Where :math:`n` is the lag.
    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.
    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    else:
        ValueError("Bad correlation mode option.")
    return lags


# NOT TESTED YET
def corr_normalized(in1, in2, sampling_interval=1, unbiased=True, normalized=True):
    ''' calculate correlation between two inputs
    Parameters
    --------
    in1 : 1st input series (should be 1-d)
    in2 : 2nd input series (in1 and in2 should be of equal length) (should be 1-d)
    sampling_interval: reciprocal of the sampling rate

    Returns
    --------
    corr_lags : correlation lags
    corr_res  : correlation results
    '''
    if len(in1.shape)!=1 or len(in2.shape)!=1 or in1.shape[-1]!=in2.shape[-1]:
        raise ValueError("")
    N = in1.shape[0]
    corr_lags = my_correlation_lags(in1.shape[0], in2.shape[0], mode="same")
    corr_res  = signal.correlate(in1, in2, mode="same")
    if normalized:
        r0_in1 = np.sum(in1*in1)
        r0_in2 = np.sum(in2*in2)
        corr_res = corr_res / np.sqrt(r0_in1*r0_in2)
    if unbiased:
        corr_res = corr_res / (-np.abs(corr_lags)+N)
        if normalized:
            corr_res = corr_res * N # so that the unbiased-normalized autocorrelation takes value 1 at lag 0. 

    return corr_lags*sampling_interval, corr_res


def resample_0phase(d, sampling_interval, up, dn, axis=0):
    """ Performs zero-phase resampling using scipy.signal.resample_poly
        Parameters
        --------
        d : data to be resampled
        sampling_interval : sampling interval of original data
        up : (integer) how many times to upsample
        dn : (integer) how many times to downsample
        axis : which axis to resample; default to 0 just as `resample_poly`
    """
    l_extrap = d.shape[axis]//8
    resampling_interval = sampling_interval * dn / up 

    # CONCATENATE WITH MIRRORING MODE
    # TODO are there ways to avoid `apply_along_axis`? (i.e. slice a particular axis without explicitly using the ':' and ','s) 
    concat_func = lambda arr: np.concatenate([arr[:l_extrap][::-1], arr, arr[-l_extrap:][::-1]])
    d_tmp = np.apply_along_axis(concat_func, axis, d)
    # print(d.shape, d_tmp.shape)
    # d_tmp = np.concatenate([d[:l_extrap][::-1], d, d[-l_extrap:][::-1]])

    l_extrap_re = int(l_extrap*up/dn) 
    # d_r = signal.resample_poly(d_tmp, up, dn)[l_extrap_re:-l_extrap_re]
    d_tmp_r = signal.resample_poly(d_tmp, up, dn, axis=axis)
    d_r = np.apply_along_axis(
        lambda x: x[l_extrap_re:-l_extrap_re],
        axis,
        d_tmp_r
    )
    # print(d_r.shape)
    return d_r, resampling_interval