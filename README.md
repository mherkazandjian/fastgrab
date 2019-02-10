# FastGrab

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
360p          | > 300
720p          | 140
1080p         | 60
4K            | 10

# Usage example

````python
  from fastgrab import screenshot
  # take a full screen screenshot
  grab = screenshot.Screenshot()
  img = grab.capture()   # img is a numpy ndarray, do whatever you want with it
````
## Getting Started

``Fastgrab`` was initially developed in 2016 as part of an aimbot (for quake
live). ``Fastgrab`` is developed and tested on ``linux``. As far as the 
pre-requisited listed below are satisfied, it should work as expected on
``windows`` and ``osx``. The low-level API of ``Fastgrab`` is implemented
using the ``cpython``, ``Numpy`` and ``X11`` C APIs.

## Comparison with other packages

The following comparison has been done a Intel i7-4770K with 32 GB ram and a
Nvidia GTX 960 at a 1080p resolution. ``Fastgrab`` is designed to be fast and
does not provide any features beyond capturing the screen, unlike the other
packages mentioned in  the comparison below that do many great things. 

package       | fps
------------- | -----
fastgrab      | 60
autopy        | 10 
pyautogui     | 1
pyscreennshot | 0.5

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

````bash
cd fastgrab
pytest tests
````

or

````bash
pip install tox-pipenv
cd fastgrab
tox
````

## Contributing

Submit a pull request or create an [issue](https://github.com/mherkazandjian/fastgrab/issues/new)
if you find any bugs.

Any help/pull requests for the following are welcome:

   - python 2.7
   - osx
   - windows

## Authors

* **Mher Kazandjian** - *Initial work* - [Github](https://github.com/mherkazandjian)

## License

This project is licensed under GPLv3

## Acknowledgments

* pyscreenshot
* autopy
* pyautogui
* reame template taken from: [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* https://stackoverflow.com/questions/69645/take-a-screenshot-via-a-python-script-linux/16141058#16141058
