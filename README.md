# FastGrab

``Fastgrab`` is an opensouce high frame rate screen capture package. A typical
capture frame rate at a resolution of 1080p on a modern machine is ~60 fps.
There are several other such packages in the wild that are opensource as well, 
but none of them is as fast or provides a simple way of obtaining the captures
image as a numpy array out of the box. The default behavior of ``fastgrab`` is
to provide the user with the image as a numpy array. Beyond that the user is
free to manipulate the image since the pixel data is accessible via a fast and
flexible array, i.e a numpy array.

capture frame rates

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
live). ``Fastgrab`` is developed and tested on ``linux``. As far was the 
pre-requisited listed below are satisfied, it should work as expected on
``windows`` and ``osx``.

## Comparison with other packages

The following comparison has been done a Intel i7-4770K with 32 GB ram and a
Nvidia GTX 960 at a 1080p resolution. 

    package

### Prerequisites

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
python setup.py install
```

## Running the tests

````bash
pytest tests
````

or 

````bash
tox
````

## Contributing

submit a pull request or create an issue if you find any bugs.

Any help/pull requests for windows/osx support are welcome.

## Authors

* **Mher Kazandjian** - *Initial work* - [Github](https://github.com/mherkazandjian)

## License

This project is licensed under GPLv3

## Acknowledgments

* pyscreenshot
* autopy
* pyautogui
* reame template taken from: [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)