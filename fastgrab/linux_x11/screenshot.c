#include <stdio.h>
#include <X11/Xlib.h>
#include <Python.h>
#include <omp.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

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
               unsigned char* numpy_data)
{
    XImage* img;
    Display* display;
    const int bytes_per_pixel = 4;

    display = XOpenDisplay(NULL);

    img = XGetImage(display,
                    RootWindow(display, DefaultScreen(display)),
                    origin_x, origin_y, width, height,
                    AllPlanes, ZPixmap);

#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        // the offset in bytes from the beginning of the buffer to the
        // data of the y^th line
        const int y_pixel_data_index_offset = y * img->bytes_per_line;
        for (int x = 0; x < width; x++)
        {
            const int pixel_data_index =  y_pixel_data_index_offset + x * bytes_per_pixel;
            const char *pixel_data = &img->data[pixel_data_index];
            int linear_index = y * width + x;

            numpy_data[bytes_per_pixel*linear_index + 0] = (unsigned int)pixel_data[2];
            numpy_data[bytes_per_pixel*linear_index + 1] = (unsigned int)pixel_data[1];
            numpy_data[bytes_per_pixel*linear_index + 2] = (unsigned int)pixel_data[0];
            numpy_data[bytes_per_pixel*linear_index + 3] = (unsigned int)0;
        }
    }

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
PyObject* linux_x11_screenshot(PyObject *self, PyObject *args)
{
    int x, y;
    long int * shape;
    PyObject *_img=NULL;
    PyArrayObject *img=NULL;

    if (!PyArg_ParseTuple(args, "iiO", &x, &y, &_img))
        return NULL;

    img = (PyArrayObject*)PyArray_FROM_OTF(_img, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (img == NULL) goto fail;

    shape = PyArray_SHAPE(img);
    screenshot(x, y, shape[1], shape[0], PyArray_DATA(img));

  fail:
    Py_DECREF(img);
    Py_RETURN_NONE;
}


// method function definitions
static PyMethodDef linux_x11_methods[] = {
    {"resolution", linux_x11_screen_resolution, METH_VARARGS, "return the screen resolution"},
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
