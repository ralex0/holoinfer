import numpy as np


def dist(*vec):
    """
    Cartesian distance formula on three arguments
    :param vec: x, y, z, 
    :return: r = sqrt(x^2 + y^2 + z^2)
    """
    return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)


def cartesian(*axes):
    """
    Cartesian product of a list of numpy arrays. Given n arrays, it returns all n-tuples generated from them as an array
    
    Parameters
    ----------
    axes : list n of numpy.ndarrays with lengths n1, n2, n3, ...

    Returns
    -------
    grid : ndarray
        shape : ((n1 * n2 * ... ), n)
    """
    return np.array(np.meshgrid(*axes, indexing='ij')).T.reshape(-1, len(axes))

