# FastGrab

[![CircleCI](https://circleci.com/gh/mherkazandjian/fastgrab/tree/master.svg?style=svg)](https://circleci.com/gh/mherkazandjian/fastgrab/tree/master)

``Fastgrab`` is an opensouce high frame rate screen capture package. A typical
capture frame rate at a resolution of 1080p on a modern machine is ~60 fps.
There are several other such packages in the wild that are opensource as well, 
but none of them is as fast or provides a simple way of obtaining the captures
image as a numpy array out of the box. The default behavior of ``fastgrab`` is
to provide the user with the image as a numpy array. Beyond that the user is
free to manipulate the image since the pixel data is accessible via a fast and
flexible array, i.e a numpy array.

Typical capture frame rate on a modern machine

resolution    | fps
------------- | -----
360p          | > 800
720p          | 260
1080p         | 200
4K            | 20

# Usage example

````python
  from fastgrab import screenshot
  # take a full screen screenshot
  img = screenshot.Screenshot().capture()
  # >> img is a numpy ndarray, do whatever you want with it
  # (optional)
  # e.g it can be displayed with matplotlib (install matplotlib first)
  from matplotlib import pyplot as plt
  plt.imshow(img[:, :, 0:3], interpolation='none', cmap='Greys_r')
  plt.show()
````
## Getting Started

``Fastgrab`` was initially developed in 2016 as part of an aimbot (for quake
live). ``Fastgrab`` is developed and tested on ``linux``. As far as the 
pre-requisited listed below are satisfied, it should work as expected on
``windows`` and ``osx``. The low-level API of ``Fastgrab`` is implemented
using the ``cpython``, ``Numpy`` and ``X11`` C APIs.

## Comparison with other packages

The following comparison has been done a Intel i7-6700HQ with 16 GB ram  at a
1080p resolution. ``fastgrab`` is designed to be fast and
does not provide any features beyond capturing the screen, unlike the other
packages mentioned in  the comparison below that do many great things. 

package       | fps
------------- | -----
fastgrab      | 200
python-mss    | 180
autopy        | 34
pyautogui     | 8
pyscreenshot  | 4

to benchmark ``fastgrab`` run the script [examples/benchmark.py](https://github.com/mherkazandjian/fastgrab/blob/master/examples/benchmark.py)

### Prerequisites

 - ``python >= 3.6`` (python 2 is not supported)
 - ``Numpy >= 1.15``
 - ``gcc >= 4.8.5``
 - ``X11 => 1.20``

note that ``fastgrab`` could work with lower versions but I have not tested it
(and probaby will not). 

### Installing

``Fastgrab`` can be installed in several ways:

```bash
pip install fastgrab
```

```bash
pip install git+git://github.com/mherkazandjian/fastgrab.git
```

```bash
git clone https://github.com/mherkazandjian/fastgrab.git
cd fastgrab
python setup.py install
```

## Running the tests

Execute the following in the source root dir

````bash
pytest tests
````

or

````bash
pip install tox-pipenv
pipenv install
tox
````

## Contributing

Submit a pull request or create an [issue](https://github.com/mherkazandjian/fastgrab/issues/new)
if you find any bugs.

Any help/pull requests that implement support for the following are welcome:

   - python 2.7
   - osx
   - windows

## Authors

* **Mher Kazandjian** - [Github](https://github.com/mherkazandjian)

## License

This project is licensed under GPLv3

## Acknowledgments

* pyscreenshot
* autopy
* pyautogui
* reame template taken from: [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* https://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux/16141058#16141058
* python-mss
