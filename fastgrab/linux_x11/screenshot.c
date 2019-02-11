#include <stdio.h>
#include <X11/Xlib.h>
#include <Python.h>
//#include <omp.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define BITS_PER_BYTE 8

void screen_resolution(int *resolution)
{
    Display *display;
    Screen *screen;

    display = XOpenDisplay(NULL);
    screen = ScreenOfDisplay(display, 0);
    resolution[0] = screen->width;
    resolution[1] = screen->height;
    XCloseDisplay(display);
}

int screenshot(const int origin_x,
               const int origin_y,
               const int width,
               const int height,
               uint8_t * data)
{
    XImage* img;
    Display* display;

    display = XOpenDisplay(NULL);

    img = XGetImage(display,
                    RootWindow(display, DefaultScreen(display)),
                    origin_x, origin_y, width, height,
                    AllPlanes, ZPixmap);

    const int nbytes = width * height * img->bits_per_pixel / BITS_PER_BYTE;

    memcpy(&data[0], &img->data[0], nbytes);

    //#pragma omp parallel shared(data, img)
    //    {
    //#pragma omp for
    //        for (int i = 0; i < nbytes; i++)
    //        {
    //            data[i] = (uint8_t)img->data[i];
    //        }
    //    }

    free(&img->data[0]);
    XCloseDisplay(display);

    return 0;
}

static
PyObject* linux_x11_screen_resolution(PyObject *self, PyObject *args)
{
    int resolution[2];
    PyObject *res_tuple = PyTuple_New(2);

    screen_resolution(&resolution[0]);

    PyTuple_SetItem(res_tuple, 0, PyLong_FromLong(resolution[0]));
    PyTuple_SetItem(res_tuple, 1, PyLong_FromLong(resolution[1]));

    return res_tuple;
}

static
PyObject* linux_x11_bytes_per_pixel(PyObject *self, PyObject *args)
{
    XImage* img;
    Display* display;

    display = XOpenDisplay(NULL);
    img = XGetImage(display,
                    RootWindow(display, DefaultScreen(display)),
                    0, 0, 1, 1,
                    AllPlanes, ZPixmap);

    free(&img->data[0]);
    XCloseDisplay(display);

    return PyLong_FromLong(img->bits_per_pixel / BITS_PER_BYTE);
}

static
PyObject* linux_x11_screenshot(PyObject *self, PyObject *args)
{
    int x, y;
    long int * shape;
    PyObject *_img=NULL;
    PyArrayObject *img=NULL;

    if (!PyArg_ParseTuple(args, "iiO", &x, &y, &_img))
        return NULL;

    img = (PyArrayObject *)PyArray_FROM_OTF(_img, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);
    if (img == NULL) goto fail;

    shape = PyArray_SHAPE(img);
    screenshot(x, y, shape[1], shape[0], (uint8_t *)PyArray_DATA(img));

  fail:
    Py_DECREF(img);
    Py_RETURN_NONE;
}


// method function definitions
static PyMethodDef linux_x11_methods[] = {
    {"resolution", linux_x11_screen_resolution, METH_VARARGS, "return the screen resolution"},
    {"bytes_per_pixel", linux_x11_bytes_per_pixel, METH_VARARGS, "return the number of bytes per pixel"},
    {"screenshot", linux_x11_screenshot, METH_VARARGS, "capture a screenshot using X11"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef _linux_x11 = {
    PyModuleDef_HEAD_INIT,
    "_linux_x11",
    "module with interface functions for capturing a screenshot",
    -1,
    linux_x11_methods
};

PyMODINIT_FUNC
PyInit__linux_x11(void)
{
    PyObject *module;
    module = PyModule_Create(&_linux_x11);
    import_array();
    return module;
}
