/* Python.h before any standard header: CPython documents that
 * requirement, and here it is load-bearing rather than stylistic.
 * pyconfig.h is what defines _POSIX_C_SOURCE / _XOPEN_SOURCE for this
 * translation unit, and glibc latches its feature-test macros in the
 * first system header it sees. Included after <limits.h> — as it used
 * to be — the macros arrive too late, the unit is built as strict ISO
 * C11 (see -std=c11 in build.py), and the POSIX getpid()/close() the
 * display cache below needs are never declared. */
#include <Python.h>

#include <limits.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
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
#define FG_ERR_GEOMETRY -7
#define FG_ERR_LOST     -8
#define FG_ERR_NOMEM    -9

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
    case FG_ERR_GEOMETRY:
        return "could not read the size of the X root window";
    case FG_ERR_LOST:
        return "the X server closed the connection; the cached display "
               "was dropped, so a later call will reconnect";
    default:
        return "unknown X11 error";
    }
}

/* ------------------------------------------------------------------ *
 * Cached display connection
 *
 * Every entry point used to run XOpenDisplay(NULL) ... XCloseDisplay()
 * around a single request, so a capture loop opened and tore down one X
 * connection per frame. That is the churn behind issue #44: connecting
 * costs an auth-file read, a socket connect and a full setup reply, all
 * of which can fail transiently on a loaded machine, and the failure
 * surfaces as "cannot open X display" on a server that is perfectly
 * healthy.
 *
 * One connection is kept per process instead. Four things have to be
 * true for that to be safe:
 *
 * 1. DISPLAY is re-read on every call and the cache is keyed on it, so a
 *    caller that repoints DISPLAY gets a connection to the display it
 *    just named rather than the one it named earlier. Unset and empty
 *    are distinct keys: XOpenDisplay(NULL) is still what actually
 *    resolves the name, so its own rules for both are preserved.
 *
 * 2. The GIL is held across the whole call, exactly as before this
 *    change -- no Py_BEGIN_ALLOW_THREADS was here and none is added.
 *    That is what makes a shared Display* safe without XInitThreads():
 *    two Python threads cannot be inside libX11 on this connection at
 *    the same time. XInitThreads() is deliberately not called. It has to
 *    run before any other Xlib call process-wide, so a library cannot
 *    honestly promise it, and calling it from an imported extension
 *    changes locking for every other Xlib user in the process. Releasing
 *    the GIL around XGetImage would be a real throughput win for
 *    multi-threaded callers, but it needs that global opt-in plus
 *    per-connection locking, and it is a separate change.
 *
 * 3. The owning pid is recorded. A connection inherited across fork()
 *    is poisoned: parent and child hold descriptors onto one socket and
 *    their requests interleave into a single protocol stream. A child
 *    therefore abandons the inherited Display and connects afresh. It
 *    must not XCloseDisplay() it -- that writes to the socket the parent
 *    is still using -- so the Display struct is leaked once and only the
 *    child's own descriptor is closed, which the parent never sees.
 *
 * 4. A dead connection cannot become a hard exit(). Xlib's default I/O
 *    error handler prints and calls exit(); with a cached connection a
 *    server that went away between two captures would meet that handler
 *    instead of the clean XOpenDisplay-returned-NULL path it used to.
 *    fg_io_error_handler() below longjmps back out instead, the dead
 *    connection is dropped, and the call is retried once on a fresh one
 *    so that a restarted server keeps behaving the way it did when every
 *    call opened its own connection.
 *
 * Nothing tears the connection down at interpreter shutdown, on purpose.
 * The repo already avoids finalizer-ordering hazards (the wlr and
 * windows backends have no __del__, and tests/_run_pytest_clean_exit.py
 * exists to os._exit past one), and an atexit hook or a module m_free
 * would put an X round trip at an unpredictable point in teardown for no
 * benefit: the kernel closes the socket and the server reaps the client.
 * ------------------------------------------------------------------ */

static Display *fg_display = NULL;
/* Copy of DISPLAY as it was when fg_display was opened. NULL means the
 * variable was unset, which is a different key from "" (set but empty). */
static char *fg_display_name = NULL;
static pid_t fg_display_pid = 0;
/* Connections opened since import. Only read by the private
 * _display_cache_info() below, which is how the tests observe that a
 * capture loop reuses one connection instead of opening one per frame. */
static unsigned long fg_display_opens = 0;

/* strdup() is POSIX and this file is compiled with -std=c11; rather than
 * depend on which feature-test macros happen to be in force, duplicate
 * the few bytes by hand. */
static char *fg_dup(const char *s)
{
    size_t n = strlen(s) + 1;
    char *copy = malloc(n);

    if (copy != NULL)
        memcpy(copy, s, n);
    return copy;
}

static int fg_display_name_matches(const char *env)
{
    if (env == NULL || fg_display_name == NULL)
        return env == fg_display_name;  /* both unset, or exactly one */
    return strcmp(env, fg_display_name) == 0;
}

/* Forget the cached connection.
 *
 * close_connection tells XCloseDisplay whether it may talk to the
 * server. It may not when the socket is shared with a parent process
 * (post-fork) or is already dead (I/O error): in both cases the protocol
 * write would either corrupt someone else's stream or re-enter the I/O
 * error path. The Display struct is then leaked -- once per fork and
 * once per server death, never per call -- and only the descriptor is
 * reclaimed. Closing our own descriptor sends nothing to the server and
 * leaves the parent's copy of the socket untouched. */
static void fg_drop_display(int close_connection)
{
    if (fg_display != NULL) {
        if (close_connection) {
            XCloseDisplay(fg_display);
        } else {
            int fd = ConnectionNumber(fg_display);
            if (fd >= 0)
                close(fd);
        }
    }
    fg_display = NULL;
    free(fg_display_name);
    fg_display_name = NULL;
    fg_display_pid = 0;
}

/* ------------------------------------------------------------------ *
 * I/O error recovery
 *
 * Installed lazily, on the first successful connect, so merely importing
 * fastgrab does not change process-global Xlib state. Re-installed on
 * every connect so that a host application which set its own handler in
 * the meantime is chained to rather than clobbered.
 * ------------------------------------------------------------------ */

static jmp_buf fg_io_jmp;
static volatile int fg_io_armed = 0;
static Display *fg_io_display = NULL;
static XIOErrorHandler fg_io_prev = NULL;

static int fg_io_error_handler(Display *dpy)
{
    if (fg_io_armed && dpy == fg_io_display) {
        fg_io_armed = 0;
        longjmp(fg_io_jmp, 1);  /* does not return */
    }
    /* Not our connection, or not inside a protected call: an I/O error
     * handler is not allowed to return, so hand it to whoever was
     * installed before us. */
    if (fg_io_prev != NULL)
        return fg_io_prev(dpy);
    fprintf(stderr, "XIO: fatal IO error on X server \"%s\"\n",
            DisplayString(dpy));
    exit(1);
}

static void fg_install_io_handler(void)
{
    XIOErrorHandler prev = XSetIOErrorHandler(fg_io_error_handler);

    if (prev != fg_io_error_handler)
        fg_io_prev = prev;
}

/* Return the cached connection, opening one if needed.
 *
 * *reused says whether the returned connection was already open, which
 * is what decides if a mid-call server death is worth retrying. */
static int fg_acquire_display(Display **out, int *reused)
{
    const char *env = getenv("DISPLAY");
    Display *dpy;
    char *name = NULL;

    *reused = 0;

    if (fg_display != NULL) {
        if (fg_display_pid != getpid())
            fg_drop_display(0);
        else if (!fg_display_name_matches(env))
            fg_drop_display(1);
    }

    if (fg_display != NULL) {
        *out = fg_display;
        *reused = 1;
        return FG_OK;
    }

    dpy = XOpenDisplay(NULL);
    if (dpy == NULL)
        return FG_ERR_DISPLAY;

    if (env != NULL) {
        name = fg_dup(env);
        if (name == NULL) {
            XCloseDisplay(dpy);
            return FG_ERR_NOMEM;
        }
    }

    fg_install_io_handler();
    fg_display = dpy;
    fg_display_name = name;
    fg_display_pid = getpid();
    fg_display_opens++;
    *out = dpy;
    return FG_OK;
}

typedef int (*fg_display_op)(Display *dpy, void *ctx);

/* Run one operation on the cached connection, catching a server that
 * dies underneath it. The setjmp target has to live in the frame that
 * is still active when the handler fires, which is this one.
 *
 * Nothing written before or after setjmp() is read on the longjmp path
 * -- it drops the connection through file statics and returns a
 * constant -- so no local needs to be volatile. Building with -Wextra
 * still reports -Wclobbered here once fg_acquire_display() is inlined:
 * that is gcc flagging a value that is live across the setjmp, not one
 * this code reads afterwards. The project's own flags (-Wall
 * -Wsign-compare) are clean. */
static int fg_try_once(fg_display_op op, void *ctx, int *reused)
{
    Display *dpy;
    int rc;

    rc = fg_acquire_display(&dpy, reused);
    if (rc != FG_OK)
        return rc;

    if (setjmp(fg_io_jmp) != 0) {
        fg_io_armed = 0;
        fg_drop_display(0);
        return FG_ERR_LOST;
    }

    fg_io_display = dpy;
    fg_io_armed = 1;
    rc = op(dpy, ctx);
    fg_io_armed = 0;
    return rc;
}

static int fg_call_with_display(fg_display_op op, void *ctx)
{
    int reused = 0;
    int rc = fg_try_once(op, ctx, &reused);

    /* A connection that was already open can have gone stale while it
     * sat idle -- the server restarted, the session ended. Before this
     * cache existed every call connected from scratch and simply saw the
     * new server, so reconnect once and repeat the request rather than
     * making callers handle a failure they never used to see. Only a
     * reused connection is retried, so a genuinely dead server cannot
     * loop, and every operation here is a read: repeating one is safe. */
    if (rc == FG_ERR_LOST && reused)
        rc = fg_try_once(op, ctx, &reused);
    return rc;
}

/* ------------------------------------------------------------------ *
 * Operations
 * ------------------------------------------------------------------ */

/* Ask the server for the root window's current size.
 *
 * Not ScreenOfDisplay()->width/height: those come from the connection
 * setup and are only refreshed by an Xlib client that processes RandR
 * events, which this one does not. With a per-call connection reading
 * them was accurate by accident; with a cached one they would freeze at
 * whatever the screen measured when fastgrab first connected. A screen
 * that then shrank would still pass the bounds check below and reach
 * XGetImage as a BadMatch -- delivered to Xlib's default error handler,
 * which exits the interpreter. One extra round trip is the price of the
 * bounds check staying true, and it keeps resolution() reporting the
 * live size the way it did before. */
static int fg_root_size(Display *dpy, int *width, int *height)
{
    Window root = RootWindow(dpy, DefaultScreen(dpy));
    Window root_return;
    int x, y;
    unsigned int w, h, border, depth;

    if (!XGetGeometry(dpy, root, &root_return, &x, &y, &w, &h,
                      &border, &depth))
        return FG_ERR_GEOMETRY;
    if (w > (unsigned int)INT_MAX || h > (unsigned int)INT_MAX)
        return FG_ERR_GEOMETRY;

    *width = (int)w;
    *height = (int)h;
    return FG_OK;
}

/* DefaultScreen(), not 0: capture and its bounds check both use
 * RootWindow(display, DefaultScreen(display)), and on a DISPLAY of the
 * form :N.1 those are different screens. Reporting one screen's size
 * while capturing another makes the high-level bbox check disagree with
 * what the server will actually allow. (A DISPLAY that names a different
 * screen is also a different cache key, so switching between them
 * reconnects rather than answering from the wrong connection.) */
static int fg_op_resolution(Display *dpy, void *ctx)
{
    int *resolution = (int *)ctx;

    return fg_root_size(dpy, &resolution[0], &resolution[1]);
}

static int screen_resolution(int *resolution)
{
    return fg_call_with_display(fg_op_resolution, resolution);
}

struct fg_shot_ctx {
    int origin_x;
    int origin_y;
    int width;
    int height;
    uint8_t *data;
    size_t capacity;
};

static int fg_op_screenshot(Display *dpy, void *ctx)
{
    struct fg_shot_ctx *req = (struct fg_shot_ctx *)ctx;
    XImage *img;
    int screen_width, screen_height;
    int rc;

    rc = fg_root_size(dpy, &screen_width, &screen_height);
    if (rc != FG_OK)
        return rc;

    /* Written as subtractions so a large origin cannot overflow int. */
    if (req->origin_x < 0 || req->origin_y < 0 ||
        req->origin_x > screen_width - req->width ||
        req->origin_y > screen_height - req->height)
        return FG_ERR_BOUNDS;

    img = XGetImage(dpy,
                    RootWindow(dpy, DefaultScreen(dpy)),
                    req->origin_x, req->origin_y, req->width, req->height,
                    AllPlanes, ZPixmap);
    if (img == NULL)
        return FG_ERR_GETIMAGE;

    /* ZPixmap on a 32-bit visual is laid out B,G,R,A on little-endian
     * hosts, which is the byte order the Python side promises. Anything
     * else would be copied as the wrong number of bytes per pixel and
     * handed back as if it were BGRA. */
    if (img->bits_per_pixel != 32) {
        XDestroyImage(img);
        return FG_ERR_DEPTH;
    }

    const size_t row_bytes = (size_t)req->width * 4;
    const size_t nbytes = row_bytes * (size_t)req->height;

    /* Defence in depth: the caller's buffer was already checked against
     * the requested shape, but the size actually copied is derived from
     * what the server returned. */
    if (nbytes > req->capacity) {
        XDestroyImage(img);
        return FG_ERR_CAPACITY;
    }
    if ((size_t)img->bytes_per_line < row_bytes) {
        XDestroyImage(img);
        return FG_ERR_STRIDE;
    }

    if ((size_t)img->bytes_per_line == row_bytes) {
        memcpy(req->data, img->data, nbytes);
    } else {
        /* The server is free to pad rows; copying straight through would
         * shear the image by the padding on every row. */
        int row;
        for (row = 0; row < req->height; row++)
            memcpy(req->data + (size_t)row * row_bytes,
                   img->data + (size_t)row * (size_t)img->bytes_per_line,
                   row_bytes);
    }

    XDestroyImage(img);
    return FG_OK;
}

static int screenshot(const int origin_x,
                      const int origin_y,
                      const int width,
                      const int height,
                      uint8_t *data,
                      const size_t capacity)
{
    struct fg_shot_ctx req;

    /* Bounds are checked here and in fg_op_screenshot rather than left
     * to the server: a region that reaches outside the root window is a
     * BadMatch, and the NULL check on XGetImage cannot catch it. X
     * protocol errors are delivered asynchronously to Xlib's *default
     * error handler*, which prints a diagnostic and calls exit() -- the
     * interpreter dies with no exception and no traceback. The public
     * API validates too, but this entry point is reachable directly, and
     * examples/low_level_api_screenshot.py promotes exactly that. */
    if (width <= 0 || height <= 0)
        return FG_ERR_BOUNDS;

    req.origin_x = origin_x;
    req.origin_y = origin_y;
    req.width = width;
    req.height = height;
    req.data = data;
    req.capacity = capacity;

    return fg_call_with_display(fg_op_screenshot, &req);
}

static int fg_op_bytes_per_pixel(Display *dpy, void *ctx)
{
    int *bpp = (int *)ctx;
    XImage *img = XGetImage(dpy,
                            RootWindow(dpy, DefaultScreen(dpy)),
                            0, 0, 1, 1,
                            AllPlanes, ZPixmap);

    if (img == NULL)
        return FG_ERR_GETIMAGE;
    *bpp = img->bits_per_pixel / BITS_PER_BYTE;
    XDestroyImage(img);
    return FG_OK;
}

static int bytes_per_pixel(int *bpp)
{
    return fg_call_with_display(fg_op_bytes_per_pixel, bpp);
}

/* ------------------------------------------------------------------ *
 * Python entry points
 * ------------------------------------------------------------------ */

static PyObject *fg_raise(int rc)
{
    if (rc == FG_ERR_NOMEM)
        return PyErr_NoMemory();
    PyErr_SetString(PyExc_RuntimeError, fg_strerror(rc));
    return NULL;
}

static PyObject *linux_x11_screen_resolution(PyObject *self, PyObject *args)
{
    int resolution[2];
    int rc = screen_resolution(resolution);
    if (rc != FG_OK)
        return fg_raise(rc);
    return Py_BuildValue("(ii)", resolution[0], resolution[1]);
}

static PyObject *linux_x11_bytes_per_pixel(PyObject *self, PyObject *args)
{
    int bpp;
    int rc = bytes_per_pixel(&bpp);
    if (rc != FG_OK)
        return fg_raise(rc);
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

    if (rc != FG_OK)
        return fg_raise(rc);
    Py_RETURN_NONE;
}

static PyObject *linux_x11_display_cache_info(PyObject *self, PyObject *args)
{
    PyObject *name;

    if (fg_display_name == NULL) {
        name = Py_None;
        Py_INCREF(name);
    } else {
        name = PyUnicode_FromString(fg_display_name);
        if (name == NULL)
            return NULL;
    }
    return Py_BuildValue("{s:O,s:N,s:k,s:l}",
                         "connected", fg_display != NULL ? Py_True : Py_False,
                         "display", name,
                         "opens", fg_display_opens,
                         "pid", (long)fg_display_pid);
}

static PyObject *linux_x11_close_display(PyObject *self, PyObject *args)
{
    if (fg_display != NULL) {
        /* Only talk to the server if this process is the one that
         * connected; a child that inherited the connection must not. */
        fg_drop_display(fg_display_pid == getpid());
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
    {"_display_cache_info", linux_x11_display_cache_info, METH_NOARGS,
     "private, for the tests: report the cached X connection as a dict "
     "of connected / display / opens / pid. 'opens' counts connections "
     "made since import, which is how a test tells connection reuse from "
     "one connection per frame."},
    {"_close_display", linux_x11_close_display, METH_NOARGS,
     "private, for the tests: drop the cached X connection. The next "
     "call reconnects."},
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
