#include <limits.h>
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
#define FG_ERR_BOUNDS   -3
#define FG_ERR_DEPTH    -4
#define FG_ERR_CAPACITY -5
#define FG_ERR_STRIDE   -6

static const char *fg_strerror(int code)
{
    switch (code) {
    case FG_ERR_DISPLAY:
        return "cannot open X display: is DISPLAY set and the X server "
               "reachable?";
    case FG_ERR_GETIMAGE:
        return "XGetImage failed: region outside the screen, or the X "
               "server refused the request";
    case FG_ERR_BOUNDS:
        return "requested region is outside the screen, or has a "
               "non-positive width or height";
    case FG_ERR_DEPTH:
        return "X server returned a non 32-bit image; fastgrab needs a "
               "32-bit ZPixmap visual to produce BGRA";
    case FG_ERR_CAPACITY:
        return "image buffer is too small for the requested region";
    case FG_ERR_STRIDE:
        return "X server returned a row stride narrower than the "
               "requested width";
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
    /* DefaultScreen(), not 0: capture and its bounds check both use
     * RootWindow(display, DefaultScreen(display)), and on a DISPLAY of
     * the form :N.1 those are different screens. Reporting one screen's
     * size while capturing another makes the high-level bbox check
     * disagree with what the server will actually allow. */
    screen = ScreenOfDisplay(display, DefaultScreen(display));
    resolution[0] = screen->width;
    resolution[1] = screen->height;
    XCloseDisplay(display);
    return FG_OK;
}

static int screenshot(const int origin_x,
                      const int origin_y,
                      const int width,
                      const int height,
                      uint8_t *data,
                      const size_t capacity)
{
    XImage *img;
    Display *display;
    Screen *screen;

    /* Bounds are checked here rather than left to the server: a region
     * that reaches outside the root window is a BadMatch, and the NULL
     * check below cannot catch it. X protocol errors are delivered
     * asynchronously to Xlib's *default error handler*, which prints a
     * diagnostic and calls exit() -- the interpreter dies with no
     * exception and no traceback. The public API validates too, but
     * this entry point is reachable directly, and
     * examples/low_level_api_screenshot.py promotes exactly that. */
    if (width <= 0 || height <= 0)
        return FG_ERR_BOUNDS;

    display = XOpenDisplay(NULL);
    if (display == NULL)
        return FG_ERR_DISPLAY;

    screen = ScreenOfDisplay(display, DefaultScreen(display));
    /* Written as subtractions so a large origin cannot overflow int. */
    if (origin_x < 0 || origin_y < 0 ||
        origin_x > screen->width - width ||
        origin_y > screen->height - height) {
        XCloseDisplay(display);
        return FG_ERR_BOUNDS;
    }

    img = XGetImage(display,
                    RootWindow(display, DefaultScreen(display)),
                    origin_x, origin_y, width, height,
                    AllPlanes, ZPixmap);
    if (img == NULL) {
        XCloseDisplay(display);
        return FG_ERR_GETIMAGE;
    }

    /* ZPixmap on a 32-bit visual is laid out B,G,R,A on little-endian
     * hosts, which is the byte order the Python side promises. Anything
     * else would be copied as the wrong number of bytes per pixel and
     * handed back as if it were BGRA. */
    if (img->bits_per_pixel != 32) {
        XDestroyImage(img);
        XCloseDisplay(display);
        return FG_ERR_DEPTH;
    }

    const size_t row_bytes = (size_t)width * 4;
    const size_t nbytes = row_bytes * (size_t)height;

    /* Defence in depth: the caller's buffer was already checked against
     * the requested shape, but the size actually copied is derived from
     * what the server returned. */
    if (nbytes > capacity) {
        XDestroyImage(img);
        XCloseDisplay(display);
        return FG_ERR_CAPACITY;
    }
    if ((size_t)img->bytes_per_line < row_bytes) {
        XDestroyImage(img);
        XCloseDisplay(display);
        return FG_ERR_STRIDE;
    }

    if ((size_t)img->bytes_per_line == row_bytes) {
        memcpy(data, img->data, nbytes);
    } else {
        /* The server is free to pad rows; copying straight through would
         * shear the image by the padding on every row. */
        int row;
        for (row = 0; row < height; row++)
            memcpy(data + (size_t)row * row_bytes,
                   img->data + (size_t)row * (size_t)img->bytes_per_line,
                   row_bytes);
    }

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

    /* Reject rather than coerce. PyArray_FROM_OTF would happily build a
     * temporary uint8 C-contiguous copy of a mismatched buffer, let the
     * capture fill *that*, and then discard it -- the caller's array
     * would come back untouched with no error raised. Propagating the
     * result instead would need NPY_ARRAY_INOUT_ARRAY2 plus
     * PyArray_ResolveWritebackIfCopy; for a buffer the caller allocates
     * specifically to be filled, a strict contract is clearer. */
    if (!PyArray_Check(_img)) {
        PyErr_SetString(PyExc_TypeError,
                        "image buffer must be a numpy ndarray");
        return NULL;
    }
    img = (PyArrayObject *)_img;   /* borrowed; do not DECREF */

    if (PyArray_TYPE(img) != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError,
                        "image buffer must have dtype uint8");
        return NULL;
    }
    if (!PyArray_ISCARRAY(img)) {
        PyErr_SetString(PyExc_ValueError,
                        "image buffer must be C-contiguous, aligned and "
                        "writable");
        return NULL;
    }
    if (PyArray_NDIM(img) != 3 || PyArray_SHAPE(img)[2] != 4) {
        PyErr_SetString(PyExc_ValueError,
                        "image buffer must be a (height, width, 4) array");
        return NULL;
    }

    shape = PyArray_SHAPE(img);
    if (shape[0] <= 0 || shape[1] <= 0 ||
        shape[0] > INT_MAX || shape[1] > INT_MAX) {
        PyErr_SetString(PyExc_ValueError,
                        "image buffer height and width must be positive "
                        "and fit in a C int");
        return NULL;
    }

    rc = screenshot(x, y, (int)shape[1], (int)shape[0],
                    (uint8_t *)PyArray_DATA(img),
                    (size_t)PyArray_NBYTES(img));

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
