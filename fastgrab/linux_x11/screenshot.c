#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define BITS_PER_BYTE 8

/* Error codes returned by the plain-C helpers below. The Python wrappers
 * translate them into RuntimeError so a missing or unreachable X server
 * surfaces as an exception instead of a NULL dereference (segfault). */
#define FG_OK            0
#define FG_ERR_DISPLAY  -1
#define FG_ERR_GETIMAGE -2

static const char *fg_strerror(int code)
{
    switch (code) {
    case FG_ERR_DISPLAY:
        return "cannot open X display: is DISPLAY set and the X server "
               "reachable?";
    case FG_ERR_GETIMAGE:
        return "XGetImage failed: region outside the screen, or the X "
               "server refused the request";
    default:
        return "unknown X11 error";
    }
}

static int screen_resolution(int *resolution)
{
    Display *display;
    Screen *screen;

    display = XOpenDisplay(NULL);
    if (display == NULL)
        return FG_ERR_DISPLAY;
    screen = ScreenOfDisplay(display, 0);
    resolution[0] = screen->width;
    resolution[1] = screen->height;
    XCloseDisplay(display);
    return FG_OK;
}

static int screenshot(const int origin_x,
                      const int origin_y,
                      const int width,
                      const int height,
                      uint8_t *data)
{
    XImage *img;
    Display *display;

    display = XOpenDisplay(NULL);
    if (display == NULL)
        return FG_ERR_DISPLAY;

    img = XGetImage(display,
                    RootWindow(display, DefaultScreen(display)),
                    origin_x, origin_y, width, height,
                    AllPlanes, ZPixmap);
    if (img == NULL) {
        XCloseDisplay(display);
        return FG_ERR_GETIMAGE;
    }

    /* One memcpy straight out of the XImage: ZPixmap on a 32-bit visual
     * is laid out B,G,R,A on little-endian hosts, which is the byte order
     * the Python side promises. */
    const size_t nbytes =
        (size_t)width * height * img->bits_per_pixel / BITS_PER_BYTE;
    memcpy(data, img->data, nbytes);

    XDestroyImage(img);
    XCloseDisplay(display);
    return FG_OK;
}

static int bytes_per_pixel(int *bpp)
{
    XImage *img;
    Display *display;

    display = XOpenDisplay(NULL);
    if (display == NULL)
        return FG_ERR_DISPLAY;
    img = XGetImage(display,
                    RootWindow(display, DefaultScreen(display)),
                    0, 0, 1, 1,
                    AllPlanes, ZPixmap);
    if (img == NULL) {
        XCloseDisplay(display);
        return FG_ERR_GETIMAGE;
    }
    *bpp = img->bits_per_pixel / BITS_PER_BYTE;
    XDestroyImage(img);
    XCloseDisplay(display);
    return FG_OK;
}

static PyObject *linux_x11_screen_resolution(PyObject *self, PyObject *args)
{
    int resolution[2];
    int rc = screen_resolution(resolution);
    if (rc != FG_OK) {
        PyErr_SetString(PyExc_RuntimeError, fg_strerror(rc));
        return NULL;
    }
    return Py_BuildValue("(ii)", resolution[0], resolution[1]);
}

static PyObject *linux_x11_bytes_per_pixel(PyObject *self, PyObject *args)
{
    int bpp;
    int rc = bytes_per_pixel(&bpp);
    if (rc != FG_OK) {
        PyErr_SetString(PyExc_RuntimeError, fg_strerror(rc));
        return NULL;
    }
    return PyLong_FromLong(bpp);
}

static PyObject *linux_x11_screenshot(PyObject *self, PyObject *args)
{
    int x, y;
    npy_intp *shape;
    PyObject *_img = NULL;
    PyArrayObject *img = NULL;
    int rc;

    if (!PyArg_ParseTuple(args, "iiO", &x, &y, &_img))
        return NULL;

    img = (PyArrayObject *)PyArray_FROM_OTF(_img, NPY_UINT8,
                                            NPY_ARRAY_OUT_ARRAY);
    if (img == NULL)
        return NULL;

    if (PyArray_NDIM(img) != 3) {
        Py_DECREF(img);
        PyErr_SetString(PyExc_ValueError,
                        "image buffer must be a (height, width, 4) array");
        return NULL;
    }

    shape = PyArray_SHAPE(img);
    rc = screenshot(x, y, (int)shape[1], (int)shape[0],
                    (uint8_t *)PyArray_DATA(img));
    Py_DECREF(img);

    if (rc != FG_OK) {
        PyErr_SetString(PyExc_RuntimeError, fg_strerror(rc));
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef linux_x11_methods[] = {
    {"resolution", linux_x11_screen_resolution, METH_VARARGS,
     "return the screen resolution"},
    {"bytes_per_pixel", linux_x11_bytes_per_pixel, METH_VARARGS,
     "return the number of bytes per pixel"},
    {"screenshot", linux_x11_screenshot, METH_VARARGS,
     "capture a screenshot using X11"},
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
    import_array();
    module = PyModule_Create(&_linux_x11);
    return module;
}
