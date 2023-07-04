<!-- omit in toc -->
# PySerialRNG: Python Interface for Serial device Random Number Generators
This package allows to use a serial device as a random number generator. In particular, it compares an array of measurements, sent from the serial port, with their moving average and from them it generates a numpy array of 0 and 1, which is then converted into unsigned integer. By using some algorithms, these are then converted into different distributions. 
This library was tested with an Arduino Due, connected to a white noise generator.

- [Setup](#setup)
  - [Setup Serial Communication](#setup-serial-communication)
  - [Setup Python](#setup-python)
- [Usage](#usage)
- [Examples](#examples)
  - [Example of ```__call__```](#example-of-__call__)
  - [Example of ```fft_test```](#example-of-fft_test)
  - [Example of ```get_uint```](#example-of-get_uint)
  - [Example of ```randint```, ```rand```, and ```normal```](#example-of-randint-rand-and-normal)
- [License](#license)
- [Open Issues](#open-issues)

## Setup
Download the git repository and unpack it in your project folder.

### Setup Serial Communication
1. A reset system should be implemented in the device in this way to be properly compatible with the package: if it receives the 'r' character, it will restart.
2. Connect the device to your computer.

### Setup Python
1. In order to use this package, you need to have installed the following packages:
    - ```pyserial```
    - ```numpy```
    - ```tqdm```
    - ```matplotlib```

    You can install them by using ```pip```:
    ```zsh
    pip install pyserial numpy tqdm matplotlib
    ```
    (The package was tested with ```Python 3.9.16```, with ```pyserial 3.5```, ```numpy 1.21.5```, ```tqdm 4.64.1``` and ```matplotlib 3.6.2```)
2. Add the following line to your code:
```python
import PySerialRNG
```

## Usage
The package can be used by calling the ```random_generator``` class, which starts the connection with the device, by using ```pyserial```. The class has the following methods:
- ```__init__```: initializes the connection with the device and sets the parameters of the random number generator. The parameter that must be passed is ```port```, which is the port to which the device is connected.
- ```__call__```: returns an array of bits of length ```num```. This method has the following parameters:
  - ```num```: the number of bits to be returned.
  - ```window_size```: the window size used to calculate the moving avergare on the measurements. The default value is 1000.
  - ```reduction```: allows to take only every n-th measure, for example if ```reduction = 4```, the function converts only ```measures[::4]```; by default ```reduction = 1```.
  - ```progress_bar```: if ```True```, a progress bar is shown. The default value is ```False```.
- ```fft_test```: returns the FFT of the input signal. In order to have good result, it should be uniform.
  - ```num```: the number of measures; by default is $10^5$.
  - ```reduction```: allows to take only every n-th measure, for example if ```reduction = 4```, the function converts only ```measures[::4]```; by default ```reduction = 1```.
- ```exit```: closes the connection with the device. Must be called at the end of the program.
- ```get_uint```: returns an array of unsigned integer. This method has the following parameters:
  - ```num```: the number of unsigned integers to be returned.
  - ```window_size```: the window size used to calculate the moving avergare on the measurements. The default value is 1000.
  - ```dtype```: the data type of the uinsigned int returned in the array. It must be 8, 16, 32 or 64. The default value is 16.
  - ```reduction```: allows to take only every n-th measure, for example if ```reduction = 4```, the function converts only ```measures[::4]```; by default ```reduction = 1```.
  - ```progress_bar```: if ```True```, a progress bar is shown. The default value is ```False```.

The ```random``` subclass of ```random_generator``` allows to obtain different number distributions, in particular it contains the following methods:
- ```randint```: returns an array of integers uniformly distributed between `low` and `high`, given by the user.
- ```rand```: returns an array of floats uniformly distributed in [0,1).
- ```normal```: returns an array of floats in a normal distribution of mean `loc` and standard deviation `scale`.

## Examples
### Example of ```__call__```
```python
import PySerialRNG as rg
import numpy as np
gen = rg.random('/dev/cu.usbmodem11101')
data = gen(int(1e5), window_size=1000, reduction=4, progress_bar=True)
print('Data acquired')
gen.exit()
```

### Example of ```fft_test```
```python
import PySerialRNG as rg
import numpy as np
gen = rg.random('/dev/cu.usbmodem11101')
gen.fft_test(int(1e6), reduction=4)
gen.exit()
```

### Example of ```get_uint```
```python
import PySerialRNG as rg
import numpy as np
gen = rg.random('/dev/cu.usbmodem11101')
data = gen.get_uint(int(1e6), window_size=1000, dtype=32, reduction=4, progress_bar=True)
print('Data acquired')
gen.exit()
```

### Example of ```randint```, ```rand```, and ```normal``` 
```python
import PySerialRNG as rg
import numpy as np
gen = rg.random('/dev/cu.usbmodem11101')
x_randint = gen.randint(10, 1000, int(1e5), reduction=4, progress_bar=True)
x_rand = gen.rand(int(1e5), reduction=4, progress_bar=True)
x_norm = gen.normal(0, 0.5, int(1e5), reduction=4, progress_bar=True)
print('Data acquired')
gen.exit()
```

## License

BSD 3-Clause License

For additional information check the provided license file.

## Open Issues
- When the device is restarting, python may raise a warning. The warning is raised by ```pyserial``` and can be ignored.