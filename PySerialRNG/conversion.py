#
# Copyright (C) 2023 Elia Falcioni
#
# PySerialRNG is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>. 

import numpy as np


def moving_average(data: np.ndarray, window_size: int) -> tuple[int, np.ndarray]:
    '''Computes the moving average of a data array with a given window size.

    Parameters
    ----------
    data : ndarray
        The data to be analyzed.
    window_size : int
        The length of the window of the moving average

    Returns
    -------
    tuple[int, np.ndarray]
        The number of elements that are cutoff and the array of the moving average
    '''
    window = np.ones(window_size) / float(window_size)
    filtred_data = np.convolve(data, window, 'same')
    bound = window_size//2
    return bound, filtred_data[bound:-bound]


def measure_to_binary(measure_array: np.ndarray, window_size: int, level: float = 1.0) -> np.ndarray:
    """Perform the conversion from measures to binary numbers. It works by firstly 
    computing the threshold using a moving average and then if the measures 
    is greater or equal to the threshold it returns 1, otherwise 0.

    Parameters
    ----------
    measure_array : np.ndarray
        The set of data to compere with the threshold 
    window_size : int
        The length of the filter window for the moving average
    level : float, optional
        Allows to move the threshold, by default 1

    Returns
    -------
    np.ndarray
        An array of bits that has the same size as input
    """
    measure_array = measure_array.reshape(-1)
    bound, threshold = moving_average(measure_array, window_size)
    threshold = threshold*level
    return np.where(measure_array[bound:-bound] >= threshold, 1, 0)


def measure_to_binary_with_average(measure_array: np.ndarray, level: float = 1.0) -> np.ndarray:
    """Perform the conversion from measures to binary numbers. It works by firstly 
    computing the threshold as the mean of the measures and then for each measure if it 
    is greater or equal to the threshold it returns 1, otherwise 0.

    Parameters
    ----------
    measure_array : np.ndarray
        The set of data to compere with the threshold 
    level : float, optional
        Allows to move the threshold, by default 1

    Returns
    -------
    np.ndarray
        An array of bits that has the same size as input
    """
    measure_array = measure_array.reshape(-1)
    threshold = np.mean(measure_array)*level
    return np.where(measure_array >= threshold, 1, 0)


def vonNeumann(data: np.ndarray) -> np.ndarray:
    """Implementation of the Von Neumann algorithm to obtain an unbiased array of bits

    Parameters
    ----------
    data : np.ndarray
        The set of data to apply the von Neumann algorithm

    Returns
    -------
    np.ndarray
        The unbiased array of bits obtained from the von Neumann algorithm
    """
    if len(data) % 2 != 0:
        data = data[:-1]
    x = data[0::2]
    y = data[1::2]
    check = np.where(x != y, x, 10)
    return check[np.where(check != 10)]


def conversion_array(data: np.ndarray, window_size: int, level: float = 1) -> np.ndarray:
    """Given a set of measures, it returns an array of bits obtained by firstly 
    performing the moving average to set the threshold and returning 1 
    if the measure is greater or equal to it, and then applying the vonNeumann algorithm
    in order to have an unbiased array of bits.

    Parameters
    ----------
    data : np.ndarray
        The set of measures to convert
    window_size : int
        The length of the filter window for the moving average
    level : float, optional
        Allows to move the threshold, by default 1

    Returns
    -------
    np.ndarray
        The unbiased array of bits obtained from the vonNeumann algorithm
    """
    bit_array = measure_to_binary(data, window_size, level)
    bit_array = bit_array.astype(int)
    bit_array = vonNeumann(bit_array)
    return bit_array


def conversion_array_with_average(data: np.ndarray) -> np.ndarray:
    """Given a set of measures, it returns an array of bits obtained by firstly 
    computing the mean to set the threshold and returning 1 
    if the measure is greater or equal to it, otherwise 0, and then applying the vonNeumann algorithm
    in order to have an unbiased array of bits.

    Parameters
    ----------
    data : np.ndarray
        The set of measures to convert

    Returns
    -------
    np.ndarray
        The unbiased array of bits obtained from the vonNeumann algorithm
    """
    bit_array = measure_to_binary_with_average(data)
    bit_array = bit_array.astype(int)
    bit_array = vonNeumann(bit_array)
    return bit_array


if __name__ == '__main__':
    """Test the functions in this module"""

    data = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1])
    print(f'data: {data}')
    print(f'Correct solution: [0 1 0 1]')
    print(f'vonNeumann(data): {vonNeumann(data)}')
