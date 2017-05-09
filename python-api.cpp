#include <fstream>
#include <boost/ref.hpp>
#include <boost/python.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include "colorize.h"
using namespace boost::python;

namespace {
    using std::runtime_error;
    AB ab_encoder;

    tuple encode_lab (PyObject *_array) {
        PyArrayObject *array((PyArrayObject *)_array);
        if (array->nd != 4) throw runtime_error("not 4d array");
        if (array->descr->type_num != NPY_FLOAT32) throw runtime_error("not float32 array");
        if (!PyArray_ISCONTIGUOUS(array)) throw runtime_error("not contiguous");
        if (array->dimensions[3] != 3) throw runtime_error("not rgb image");
        npy_intp batch = array->dimensions[0];
        npy_intp H = array->dimensions[1];
        npy_intp W = array->dimensions[2];
        cv::Mat input(batch * H, W, CV_32FC3, array->data);
        input *= 1.0/255.0;
        cv::cvtColor(input, input, CV_BGR2Lab);

        npy_intp L_dims[] = {batch, H, W, 1};
        npy_intp AB_dims[] = {batch, H, W, AB::BINS};
        npy_intp Ws_dims[] = {batch, H, W, 1};
        PyArrayObject *L = (PyArrayObject*)PyArray_SimpleNew(4, L_dims, NPY_FLOAT32);
        PyArrayObject *AB = (PyArrayObject*)PyArray_SimpleNew(4, AB_dims, NPY_FLOAT32);
        PyArrayObject *Ws = (PyArrayObject*)PyArray_SimpleNew(4, Ws_dims, NPY_FLOAT32);
        if (!PyArray_ISCONTIGUOUS(L)) throw runtime_error("L not contiguous");
        if (!PyArray_ISCONTIGUOUS(AB)) throw runtime_error("AB not contiguous");
        if (!PyArray_ISCONTIGUOUS(W)) throw runtime_error("W not contiguous");
        auto lab = input.ptr<float const>(0);
        auto l = reinterpret_cast<float *>(L->data);
        auto ab = reinterpret_cast<float *>(AB->data);
        auto w = reinterpret_cast<float *>(Ws->data);
        unsigned total = batch * H * W;
        AB::dists_buffer_t dists;
        for (unsigned i = 0; i < total; ++i) {
            l[0] = lab[0];
            ab_encoder.encode(lab + 1, ab, w, dists);
            lab += 3;
            l += 1;
            ab += AB::BINS;
            w += 1;
        }
        return make_tuple(object(boost::python::handle<>((PyObject*)L)),
                          object(boost::python::handle<>((PyObject*)AB)),
                          object(boost::python::handle<>((PyObject*)Ws)));
    }
}

BOOST_PYTHON_MODULE(_picpac)
{
    scope().attr("__doc__") = "pic2pic C++ code";
    def("encode_lab", ::encode_lab);
}

