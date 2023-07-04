#
# Copyright (C) 2023 Elia Falcioni
#
# PySerialRNG is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

import time
import threading
import serial
import numpy as np


class SerialReader(threading.Thread):
    # Inspired by http://forum.arduino.cc/index.php?topic=137635.msg1270996#msg1270996
    """ Defines a thread for reading and buffering serial data.
    By default, about 5MSamples are stored in the buffer.
    Data can be retrieved from the buffer by calling get(N)"""

    def __init__(self, port, chunkSize=1024, chunks=5000):
        threading.Thread.__init__(self)
        # circular buffer for storing serial data until it is
        # fetched by the GUI
        self.buffer = np.zeros(chunks*chunkSize, dtype=np.uint16)

        self.chunks = chunks        # number of chunks to store in the buffer
        self.chunkSize = chunkSize  # size of a single chunk (items, not bytes)
        # pointer to most (recently collected buffer index) + 1
        self.ptr = 0
        self.port = port            # serial port handle
        self.sps = 0.0              # holds the average sample acquisition rate
        self.exitFlag = False
        self.exitMutex = threading.Lock()
        self.dataMutex = threading.Lock()

    def run(self):
        exitMutex = self.exitMutex
        dataMutex = self.dataMutex
        buffer = self.buffer
        port = self.port
        count = 0
        sps = None
        lastUpdate = time.time()

        while True:
            # see whether an exit was requested
            with exitMutex:
                if self.exitFlag:
                    break

            # read one full chunk from the serial port
            data = port.read(self.chunkSize*2)
            # convert data to 16bit int numpy array
            data = np.frombuffer(data, dtype=np.uint16)

            if np.all((data >= 0) & (data <= 2**12)) == True:
                # keep track of the acquisition rate in samples-per-second
                count += self.chunkSize
                now = time.time()
                dt = now-lastUpdate
                if dt > 1.0:
                    # sps is an exponential average of the running sample rate measurement
                    if sps is None:
                        sps = count / dt
                    else:
                        sps = sps * 0.9 + (count / dt) * 0.1
                    count = 0
                    lastUpdate = now

                # write the new chunk into the circular buffer
                # and update the buffer pointer
                with dataMutex:
                    buffer[self.ptr:self.ptr+self.chunkSize] = data
                    self.ptr = (self.ptr + self.chunkSize) % buffer.shape[0]
                    if sps is not None:
                        self.sps = sps

    def get(self, num, downsample=1):
        """ Return a tuple (time_values, voltage_values, rate)
          - voltage_values will contain the *num* most recently-collected samples
            as a 32bit float array.
          - time_values assumes samples are collected at 1MS/s
          - rate is the running average sample rate.
        If *downsample* is > 1, then the number of values returned will be
        reduced by averaging that number of consecutive samples together. In
        this case, the voltage array will be returned as 32bit float.
        """
        with self.dataMutex:  # lock the buffer and copy the requested data out
            ptr = self.ptr
            if ptr-num < 0:
                data = np.empty(num, dtype=np.uint16)
                data[:num-ptr] = self.buffer[ptr-num:]
                data[num-ptr:] = self.buffer[:ptr]
            else:
                data = self.buffer[self.ptr-num:self.ptr].copy()
            rate = self.sps

        # Convert array to float and rescale to voltage.
        # Assume 3.3V / 12bits
        # (we need calibration data to do a better job on this)
        data = data.astype(np.float32) * (3.3 / 2**12)
        if downsample > 1:  # if downsampling is requested, average N samples together
            data = data.reshape(int(num/downsample), downsample).mean(axis=1)
            num = data.shape[0]
            return np.linspace(0, (num-1)*1e-6*downsample, num), data, rate
        else:
            return np.linspace(0, (num-1)*1e-6, num), data, rate

    def reset(self):
        """Reset the device by sending the character 'r' over the serial port, 
        which activates a function in the sketch that resets the Due."""
        print('Resetting the device...')
        self.port.write(b'r')

    def exit(self):
        """ Instruct the serial thread to exit."""
        with self.exitMutex:
            self.exitFlag = True


class random_generator():
    """ A class that generates random numbers from the
    serial port. The serial port should be connected to the
    source of random numbers. At the end of the program, the exit() method must be called to close 
    the serial port and the thread that takes the measures."""

    def __init__(self, port: str):
        self.port_str = port
        self.port = serial.Serial(port)
        self.thread = SerialReader(self.port)
        self.thread.start()
        time.sleep(6)  # Waits for the device to start
        print('Device connected')
        self.counter_errors = 0

    def get_data(self, num: int, window_size: int, reduction: int) -> np.ndarray:
        """Gets the array of measures from the device. Returns an array of random bits.

        Parameters
        ----------
        num : int
            number of measures
        window_size : int
            length of window of the mobile average
        reduction : int
            allows to take only every n-th measure

        Returns
        -------
        np.ndarray
            array of random numbers
        """
        from .conversion import conversion_array, conversion_array_with_average

        time_mis, data_mis, rate_mis = self.thread.get(int(num), downsample=1)
        if num < window_size:  # Check that the number of measures is greater than the window size
            raise ValueError(
                f'The number of measures must be greater than {window_size}')
        if num//window_size < 100 or window_size < 10:
            return conversion_array_with_average(data_mis)
        bit_array = conversion_array(
            data_mis[::reduction], window_size=window_size, level=1)

        # Check that bit_array is not empty or that it has all 0 or all 1, in which case relaunch the function, if the problem persists restart the device
        if self.counter_errors > 4:
            print('The number of iterations is too high. Check that the random number generator is connected correctly. \nThe device will be restarted.\n')
            self.thread.reset()
            self.exit()
            time.sleep(5)
            self.counter_errors = 0
            self.__init__(self.port_str)

        if len(bit_array) == 0 or np.all(bit_array == 0) or np.all(bit_array == 1):
            self.counter_errors += 1
            return self.get_data(num, window_size, reduction)

        return bit_array

    def __call__(self, size: int, window_size: int = 1000, reduction: int = 1, progress_bar: bool = False) -> np.ndarray:
        from tqdm import tqdm
        '''Returns the array of the random bits of length `size`.

        Parameters
        ----------
        size : int
            length of the array of random bits
        window_size : int, optional
            length of window of the moving average, by default 1000
        reduction : int, optional
            allows to take only every n-th measure, by default 1
        progress_bar : bool, optional
            if True, it shows the progress bar, by default False

        Returns
        -------
        np.ndarray
            Array of random bits of length `size`
        '''
        count_length = 0
        data = list()
        size_request_data = size * 100/6    # According to my tests only about 6% of the measures are converted in actual random bits due to the Von Neumann algorithm
        if size_request_data > 1e6:
            size_request_data = 1e6

        if progress_bar:
            pbar = tqdm(total=size)

        while count_length < size:
            tmp = self.get_data(int(size_request_data), window_size, reduction)
            count_length += len(tmp)
            data.append(tmp)
            if progress_bar:
                pbar.update(len(tmp))  # type: ignore

        if progress_bar:
            pbar.close()  # type: ignore

        data = np.concatenate((*data,))
        diff = int(len(data)-size)
        if diff == 0:
            return data
        else:
            return data[:-diff]

    def fft_test(self, num: int = int(1e5), reduction: int = 1) -> None:
        '''Test the FFT of the source of the random number generator.

        Parameters
        ----------
        num : int, optional
            number of measures, by default int(1e5)
        reduction : int, optional
            allows to take only every n-th measure, by default 1
        '''
        import matplotlib.pyplot as plt
        time_mis, data, rate_mis = self.thread.get(num, downsample=1)

        # Check that bit_array is not empty or that it has all 0 or all 1, in which case relaunch the function, if the problem persists restart the device
        if self.counter_errors > 4:
            print('The number of iterations is too high. Check that the random number generator is connected correctly and restart the device.\n')
            self.thread.reset()
            self.exit()
            time.sleep(5)
            self.counter_errors = 0
            self.__init__(self.port_str)

        if len(data) == 0 or np.all(data == 0):
            self.counter_errors += 1
            return self.fft_test(num, reduction)

        data = data[::reduction]
        N = len(data)  # number of samples
        sr = rate_mis    # sampling rate
        T = time_mis[-1]/sr  # sampling time
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        data_fft = np.fft.fft(data[0:])
        data_fft = np.abs(data_fft)**2

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(xf[10:]*1e-3, data_fft[10:len(data_fft)//2])
        ax.set_yscale('log')
        ax.set_xlabel('f [kHz]')
        ax.set_ylabel(r'PSD $[V^2/Hz]$')
        ax.set_title('FFT')
        ax.set_xlim(xf[10]*1e-3, xf[-1]*1e-3)
        ax.grid(True, 'both', 'both', ls=':')
        # plt.savefig('fft1.png', dpi=400,
        #             transparent=False, bbox_inches="tight")
        plt.show()

    def get_uint(self, size: int, window_size: int = 1000, dtype: int = 16, reduction: int = 1, progress_bar: bool = False) -> np.ndarray:
        '''Returns an array of uint numbers of length `size`.

        Parameters
        ----------
        size : int
            length of the array of random bits
        window_size : int, optional
            length of window of the mobile_average, by default 1000
        dtype : int, optional
            type of uint, it must be 8, 16, 32 or 64, by default 16
        reduction : int, optional
            allows to take only every n-th measure, by default 1
        progress_bar : bool, optional
            if True, it shows the progress bar, by default False

        Returns
        -------
        np.ndarray
            Array of random uint of length `size`

        '''
        if dtype not in [8, 16, 32, 64]:
            raise ValueError(
                'dtype must be 8, 16, 32 or 64')
        size = int(size)*dtype
        binary_array = self.__call__(
            size, window_size=window_size, reduction=reduction, progress_bar=progress_bar)
        # print('Done acquisition')
        binary_array = binary_array.reshape(-1)
        try:
            binary_array = binary_array.reshape(
                len(binary_array)//dtype, dtype)
        except ValueError as e:
            raise ValueError(
                "Invalid array passed. Unable to resize the vector.") from e

        # Calculate the value of uint64 for each row of the binary array
        uint_array = np.sum(binary_array[:, 1:] * (
            np.power(2, np.arange(dtype-1)[::-1])), axis=1)+binary_array[:, 0]*2**(dtype-1)
        return uint_array

    def exit(self):
        """ Instruct the serial thread to exit.
        MUST BE INSERED AT THE END OF THE PROGRAM!!!"""
        try:
            self.thread.exit()
            time.sleep(1)
            self.port.close()
        except serial.SerialException:
            pass


class random(random_generator):
    """ A class that generates random numbers from the
    serial port. The serial port should be connected to the
    source of random numbers. At the end of the program, the exit() method must be called to close 
    the serial port and the thread that takes the measures."""

    def __init__(self, port: str):
        super().__init__(port)

    def randint(self, low: int, high: int, size: int, accuracy: int = 16, window_size: int = 1000, reduction: int = 1, progress_bar: bool = False) -> np.ndarray:
        '''Return random integers from `low` (inclusive) to `high` (inclusive).

        Parameters
        ----------
        low : int
            lowest (signed) integers to be drawn from the distribution
        high : int
            largest (signed) integer to be drawn from the distribution
        size : int
            length of the array of random integers
        accuracy : int, optional
            type of uint that the source generates, it must be 8, 16, 32 or 64, by default 16
        window_size : int, optional
            length of window of the mobile_average, by default 1000
        reduction : int, optional
            allows to take only every n-th measure, by default 1
        progress_bar : bool, optional
            if True, it shows the progress bar, by default False

        Returns
        -------
        np.ndarray
            an array of random integers of length `size` between `low` and `high`
        '''
        data = self.get_uint(size, window_size, accuracy,
                             reduction, progress_bar)
        return np.interp(data, (0, 2**accuracy-1), (low, high)).astype(int)

    def rand(self, size: int, accuracy: int = 16, window_size: int = 1000, reduction: int = 1, progress_bar: bool = False) -> np.ndarray:
        '''Create an array of the given shape and populate it with random 
        samples from a uniform distribution over [0, 1).

        Parameters
        ----------
        size : int
            length of the output array
        accuracy : int, optional
            type of uint that the source generates, it must be 8, 16, 32 or 64, by default 16
        window_size : int, optional
            length of window of the mobile_average, by default 1000
        reduction : int, optional
            allows to take only every n-th measure, by default 1
        progress_bar : bool, optional
            if True, it shows the progress bar, by default False

        Returns
        -------
        np.ndarray
            an array of random floats of length `size` between 0 and 1 uniformly distributed
        '''
        data = self.get_uint(
            size, window_size, dtype=accuracy, reduction=reduction)
        return data/(2**accuracy)

    def normal(self, loc: float, scale: float, size: int, accuracy: int = 16, window_size: int = 1000, reduction: int = 1, progress_bar: bool = False) -> np.ndarray:
        '''
        Draw random samples in a normal (Gaussian) distribution. It gets two 
        uniform distributions and convert them in a normal distribution of mean `loc` 
        and standard deviation `scale`.

        Parameters
        ----------
        loc : float
            mean ("centre") of the distribution.
        scale : float
            standard deviation (spread or "width") of the distribution. Must be non-negative.
        size : int
            length of the output array
        window_size : int, optional
            length of window of the mobile_average, by default 1000
        reduction : int, optional
            allows to take only every n-th measure, by default 1
        progress_bar : bool, optional
            if True, it shows the progress bar, by default False

        Returns
        -------
        np.ndarray
            an array of random floats of length `size`
        '''
        U1 = self.rand(size, accuracy, window_size, reduction, progress_bar)
        U2 = self.rand(size, accuracy, window_size, reduction, progress_bar)
        R = np.sqrt(-2 * (scale**2) * np.log(1-U1))  # type: ignore
        Theta = 2 * np.pi * U2  # type: ignore
        X = R * np.cos(Theta) + loc
        return X
