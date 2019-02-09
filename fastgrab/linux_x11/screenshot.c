#include <stdio.h>
#include <X11/Xlib.h>
#include <Python.h>
#include <omp.h>

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

    /* copy to the numpy buffer using memcpy */
    //memcpy(numpy_data, img->data, width * height * sizeof(char) * 4);

    /* copy to the numpy buffer using a loop and using double so that the
     * compiler will use avx(2) instructions */
    /*
    double* from = (double *)img->data;
    double* to = (double *)numpy_data;
#pragma omp parallel for
    for( int i = 0; i < width * height / 2; i++)
    {
        to[i] = from[i];
    }
    */

    XCloseDisplay(display);

    return 0;
}

// method definitions. setting the method names and argument types and description (docstrings)
static PyMethodDef linux_x11_methods[] = {
        //"PythonName"      C-function Name      argument presentation    description
        {NULL,              NULL,                    0,                       NULL}  // sentinal
};


PyMODINIT_FUNC initlinux_x11(void)
{
    PyObject *module;
    module = Py_InitModule("linux_x11", linux_x11_methods);
    if ( module == NULL )
        return;
}


