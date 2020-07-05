import numpy as np
import cupy
from cupyx.scipy.signal.signaltools import choose_conv_method

def _iterable_of_int(x, name=None):
    """Convert ``x`` to an iterable sequence of int
    Parameters
    ----------
    x : value, or sequence of values, convertible to int
    name : str, optional
        Name of the argument being converted, only used in the error message
    Returns
    -------
    y : ``List[int]``
    """
    if isinstance(x, Number):
        x = (x,)

    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or "value"
        raise ValueError("{} must be a scalar or iterable of integers"
                         .format(name)) from e

    return x

def _init_nd_shape_and_axes(x, shape, axes):
    """Handles shape and axes arguments for nd transforms"""
    noshape = shape is None
    noaxes = axes is None

    if not noaxes:
        axes = _iterable_of_int(axes, 'axes')
        axes = [a + x.ndim if a < 0 else a for a in axes]

        if any(a >= x.ndim or a < 0 for a in axes):
            raise ValueError("axes exceeds dimensionality of input")
        if len(set(axes)) != len(axes):
            raise ValueError("all axes must be unique")

    if not noshape:
        shape = _iterable_of_int(shape, 'shape')

        if axes and len(axes) != len(shape):
            raise ValueError("when given, axes and shape arguments"
                             " have to be of the same length")
        if noaxes:
            if len(shape) > x.ndim:
                raise ValueError("shape requires more axes than are present")
            axes = range(x.ndim - len(shape), x.ndim)

        shape = [x.shape[a] if s == -1 else s for s, a in zip(shape, axes)]
    elif noaxes:
        shape = list(x.shape)
        axes = range(x.ndim)
    else:
        shape = [x.shape[a] for a in axes]

    if any(s < 1 for s in shape):
        raise ValueError(
            "invalid number of data points ({0}) specified".format(shape))

    return shape, axes


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """Determine if inputs arrays need to be swapped in `"valid"` mode.
    If in `"valid"` mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every calculated dimension.
    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.
    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
    if mode != 'valid':
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    if not (ok1 or ok2):
        raise ValueError("For 'valid' mode, one must be at least "
                         "as large as the other in every dimension")

    return not ok1

def _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    """Handle the axes argument for frequency-domain convolution.
    Returns the inputs and axes in a standard form, eliminating redundant axes,
    swapping the inputs if necessary, and checking for various potential
    errors.
    Parameters
    ----------
    in1 : array
        First input.
    in2 : array
        Second input.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the FFTs.
    sorted_axes : bool, optional
        If `True`, sort the axes.
        Default is `False`, do not sort.
    Returns
    -------
    in1 : array
        The first input, possible swapped with the second input.
    in2 : array
        The second input, possible swapped with the first input.
    axes : list of ints
        Axes over which to compute the FFTs.
    """
    s1 = in1.shape
    s2 = in2.shape
    noaxes = axes is None

    _, axes = _init_nd_shape_and_axes(in1, shape=None, axes=axes)

    if not noaxes and not len(axes):
        raise ValueError("when provided, axes cannot be empty")

    # Axes of length 1 can rely on broadcasting rules for multipy,
    # no fft needed.
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]

    if sorted_axes:
        axes.sort()

    if not all(s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1
               for a in range(in1.ndim) if a not in axes):
        raise ValueError("incompatible shapes for in1 and in2:"
                         " {0} and {1}".format(s1, s2))

    # Check that input sizes are compatible with 'valid' mode.
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        # Convolution is commutative; order doesn't have any effect on output.
        in1, in2 = in2, in1

    return in1, in2, axes

def _freq_domain_conv(in1, in2, axes, shape):
    """Convolve two arrays in the frequency domain.
    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.
    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.
    """
    if not len(axes):
        return in1 * in2

    complex_result = (in1.dtype.kind == 'c' or in2.dtype.kind == 'c')
    fshape = shape

    if not complex_result:
        fft, ifft = cupy.fft.rfftn, cupy.fft.irfftn
    else:
        fft, ifft = cupy.fft.fftn, cupy.fft.ifftn

    sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)
    ret = ifft(sp1 * sp2, fshape, axes=axes)
    return ret

def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _apply_conv_mode(ret, s1, s2, mode, axes):
    """Calculate the convolution result shape based on the `mode` argument.
    Returns the result sliced to the correct size for the given mode.
    Parameters
    ----------
    ret : array
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.
    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.
    """
    if mode == "full":
        return ret.copy()
    elif mode == "same":
        return _centered(ret, s1).copy()
    elif mode == "valid":
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                       for a in range(ret.ndim)]
        return _centered(ret, shape_valid).copy()
    else:
        raise ValueError("acceptable mode flags are 'valid',"
                         " 'same', or 'full'")


def fftconvolve(in1, in2, mode="full", axes=None):
    in1 = cupy.asarray(in1)
    in2 = cupy.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2 
    elif not in1.ndim == in2.ndim:
        raise ValueError("Dimensions do not match.")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return cupy.array([])
    try:
        from scipy.signal.signaltools import _init_freq_conv_axes
    except:
        ...
    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    s1 = in1.shape
    s2 = in2.shape

    shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
             for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, axes, shape)
    try:
        from scipy.signal.signaltools import _apply_conv_mode
    except:
        ...
    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """Determine if inputs arrays need to be swapped in `"valid"` mode.
    If in `"valid"` mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every calculated dimension.
    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.
    Note that if the mode provided is not 'valid', False is immediately
    returned.
    """
    if mode != 'valid':
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    if not (ok1 or ok2):
        raise ValueError("For 'valid' mode, one must be at least "
                         "as large as the other in every dimension")

    return not ok1


def convolve(in1, in2, mode='full', method='auto'):
    """
    Convolve two N-dimensional arrays.
    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.
    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.
        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See Notes for more detail.
           .. versionadded:: 0.19.0
    Returns
    -------
    convolve : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
    See Also
    --------
    numpy.polymul : performs polynomial multiplication (same operation, but
                    also accepts poly1d objects)
    choose_conv_method : chooses the fastest appropriate convolution method
    fftconvolve : Always uses the FFT method.
    oaconvolve : Uses the overlap-add method to do convolution, which is
                 generally faster when the input arrays are large and
                 significantly different in size.
    Notes
    -----
    By default, `convolve` and `correlate` use ``method='auto'``, which calls
    `choose_conv_method` to choose the fastest method using pre-computed
    values (`choose_conv_method` can also measure real-world timing with a
    keyword argument). Because `fftconvolve` relies on floating point numbers,
    there are certain constraints that may force `method=direct` (more detail
    in `choose_conv_method` docstring).
    Examples
    --------
    Smooth a square pulse using a Hann window:
    >>> from scipy import signal
    >>> sig = np.repeat([0., 1., 0.], 100)
    >>> win = signal.hann(50)
    >>> filtered = signal.convolve(sig, win, mode='same') / sum(win)
    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('Original pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> ax_win.plot(win)
    >>> ax_win.set_title('Filter impulse response')
    >>> ax_win.margins(0, 0.1)
    >>> ax_filt.plot(filtered)
    >>> ax_filt.set_title('Filtered signal')
    >>> ax_filt.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()
    """
    volume = cupy.asarray(in1)
    kernel = cupy.asarray(in2)

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    elif volume.ndim != kernel.ndim:
        raise ValueError("volume and kernel should have the same "
                         "dimensionality")

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative; order doesn't have any effect on output
        volume, kernel = kernel, volume

    if method == 'auto':
        method = choose_conv_method(volume, kernel, mode=mode)
    if method == 'fft':
        out = fftconvolve(volume, kernel, mode=mode)
        result_type = cupy.result_type(volume, kernel)
        if result_type.kind in {'u', 'i'}:
            out = cupy.around(out)
        return out.astype(result_type)
    elif method == 'direct':
        # fastpath to faster numpy.convolve for 1d inputs when possible
        if _np_conv_ok(volume, kernel, mode):
            return np.convolve(volume, kernel, mode)

        return correlate(volume, _reverse_and_conj(kernel), mode, 'direct')
    else:
        raise ValueError("Acceptable method flags are 'auto',"
                         " 'direct', or 'fft'.")
