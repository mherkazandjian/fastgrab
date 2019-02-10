# FastGrab

``Fastgrab`` is an opensouce high frame rate screen capture package. A typical
capture frame rate at a resolution of 1080p on a modern machine is ~60 fps.
There are several other such packages in the wild that are opensource as well, 
but none of them is as fast or provides a simple way of obtaining the captures
image as a numpy array. The default behavior of ``fastgrab`` is to provide the
user with the image as a numpy array. Beyond that the user is free to manipulate
the image since the pixel data is accessible via a fast and flexible array, i.e
a numpy array.

## Getting Started

...

### Prerequisites

 - ``Numpy``
 - ``gcc``
 - ``X11``

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

## Authors

* **Mher Kazandjian** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

## License

This project is licensed under GPLv3

## Acknowledgments

* autopy
* pyautogui
* opencv
