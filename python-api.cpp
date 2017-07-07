#include <fstream>
#include <iostream>
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
    using std::cerr;
    using std::endl;
    AB ab_encoder;

    tuple encode_bgr (PyObject *_array, int downsize) {
        PyArrayObject *array((PyArrayObject *)_array);
        if (array->nd != 4) throw runtime_error("not 4d array");
        if (array->descr->type_num != NPY_FLOAT32) throw runtime_error("not float32 array");
        if (!PyArray_ISCONTIGUOUS(array)) throw runtime_error("not contiguous");
        if (array->dimensions[3] != 3) throw runtime_error("not rgb image");
        npy_intp batch = array->dimensions[0];
        npy_intp H = array->dimensions[1];
        npy_intp W = array->dimensions[2];
        if (H % downsize != 0) throw runtime_error("H not divisible by down-size");
        if (W % downsize != 0) throw runtime_error("H not divisible by down-size");
        npy_intp oH = H / downsize;
        npy_intp oW = W / downsize;

        //cerr << batch << ':' << H << ':' << W << ':' << endl;

        npy_intp GRAY_dims[] = {batch, H, W, 1};
        npy_intp BGR_dims[] = {batch, oH, oW, 3};
        npy_intp Ws_dims[] = {batch, oH, oW, 1};
        PyArrayObject *GRAY = (PyArrayObject*)PyArray_SimpleNew(4, GRAY_dims, NPY_FLOAT32);
        PyArrayObject *BGR = (PyArrayObject*)PyArray_ZEROS(4, BGR_dims, NPY_FLOAT32, 0);
        PyArrayObject *Ws = (PyArrayObject*)PyArray_SimpleNew(4, Ws_dims, NPY_FLOAT32);
        if (!PyArray_ISCONTIGUOUS(GRAY)) throw runtime_error("GRAY not contiguous");
        if (!PyArray_ISCONTIGUOUS(BGR)) throw runtime_error("BGR not contiguous");
        if (!PyArray_ISCONTIGUOUS(Ws)) throw runtime_error("W not contiguous");

        Py_BEGIN_ALLOW_THREADS
        {   // generate L channel
            cv::Mat input(batch * H, W, CV_32FC3, array->data);
            cv::Mat output(batch * H, W, CV_32F, GRAY->data);
            cv::cvtColor(input, output, CV_BGR2GRAY);
            if ((void *)output.data != (void *)GRAY->data) throw runtime_error("cvtColor should be inplace");
        }

        {   // generate ab and w
            auto input = reinterpret_cast<char *>(array->data);
            auto bgr = reinterpret_cast<char *>(BGR->data);
            for (unsigned b = 0; b < batch; ++b) {
                cv::Mat up(H, W, CV_32FC3, input);
                cv::Mat down(oH, oW, CV_32FC3, bgr);
                cv::resize(up, down, cv::Size(oW, oH));
                input += array->strides[0];
                bgr += BGR->strides[0];
            }
            //TODO set W
        }
        Py_END_ALLOW_THREADS

        return make_tuple(object(boost::python::handle<>((PyObject*)GRAY)),
                          object(boost::python::handle<>((PyObject*)BGR)),
                          object(boost::python::handle<>((PyObject*)Ws)));
    }

    tuple encode_lab (PyObject *_array, int downsize) {
        PyArrayObject *array((PyArrayObject *)_array);
        if (array->nd != 4) throw runtime_error("not 4d array");
        if (array->descr->type_num != NPY_FLOAT32) throw runtime_error("not float32 array");
        if (!PyArray_ISCONTIGUOUS(array)) throw runtime_error("not contiguous");
        if (array->dimensions[3] != 3) throw runtime_error("not rgb image");
        npy_intp batch = array->dimensions[0];
        npy_intp H = array->dimensions[1];
        npy_intp W = array->dimensions[2];
        if (H % downsize != 0) throw runtime_error("H not divisible by down-size");
        if (W % downsize != 0) throw runtime_error("H not divisible by down-size");
        npy_intp oH = H / downsize;
        npy_intp oW = W / downsize;

        {
            cv::Mat input(batch * H, W, CV_32FC3, array->data);
            input *= 1.0/255.0;
            cv::cvtColor(input, input, CV_BGR2Lab);
            if ((void *)input.data != (void *)array->data) throw runtime_error("cvtColor should be inplace");
        }

        //cerr << batch << ':' << H << ':' << W << ':' << endl;

        npy_intp L_dims[] = {batch, H, W, 1};
        npy_intp AB_dims[] = {batch, oH, oW, AB::BINS};
        npy_intp Ws_dims[] = {batch, oH, oW, 1};
        PyArrayObject *L = (PyArrayObject*)PyArray_SimpleNew(4, L_dims, NPY_FLOAT32);
        PyArrayObject *AB = (PyArrayObject*)PyArray_ZEROS(4, AB_dims, NPY_FLOAT32, 0);
        PyArrayObject *Ws = (PyArrayObject*)PyArray_SimpleNew(4, Ws_dims, NPY_FLOAT32);
        if (!PyArray_ISCONTIGUOUS(L)) throw runtime_error("L not contiguous");
        if (!PyArray_ISCONTIGUOUS(AB)) throw runtime_error("AB not contiguous");
        if (!PyArray_ISCONTIGUOUS(Ws)) throw runtime_error("W not contiguous");

        Py_BEGIN_ALLOW_THREADS
        {   // generate L channel
            auto lab = reinterpret_cast<float *>(array->data);
            auto l = reinterpret_cast<float *>(L->data);
            unsigned total = batch * H * W;
            for (unsigned i = 0; i < total; ++i) {
                l[0] = lab[0];
                lab += 3;
                l += 1;
            }
        }

        auto ab = reinterpret_cast<float *>(AB->data);
        auto w = reinterpret_cast<float *>(Ws->data);
        AB::dists_buffer_t dists;
        {   // generate ab and w
            auto lab = reinterpret_cast<char *>(array->data);
            cv::Mat down;
            for (unsigned b = 0; b < batch; ++b) {
                cv::Mat up(H, W, CV_32FC3, lab);
                cv::resize(up, down, cv::Size(oW, oH));
                for (int i = 0; i < oH; ++i) {
                    float const *r = down.ptr<float const>(i);
                    for (unsigned j = 0; j < oW; ++j) {
                        ab_encoder.encode(r+1, ab, w, dists);
                        r += 3;
                        ab += AB::BINS;
                        w += 1;
                    }
                }
                // TODO!!!
                lab += array->strides[0];
            }
        }
        Py_END_ALLOW_THREADS

        return make_tuple(object(boost::python::handle<>((PyObject*)L)),
                          object(boost::python::handle<>((PyObject*)AB)),
                          object(boost::python::handle<>((PyObject*)Ws)));
    }

    object ab_dict () {
        npy_intp dims[] = {AB::BINS, 2};
        PyArrayObject *dict = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
        if (!PyArray_ISCONTIGUOUS(dict)) throw runtime_error("dict not contiguous");
        size_t total = 2 * AB::BINS;
        float const *from = &AB::CC[0][0];
        if (&AB::CC[AB::BINS][0] - from != total) throw runtime_error("CC not contiguous");
        std::copy(from, from+total, (float *)dict->data);
        return object(boost::python::handle<>((PyObject *)dict));
    }

    int ab_bins () {
        return AB::BINS;
    }
}

BOOST_PYTHON_MODULE(_pic2pic)
{
    import_array();
    scope().attr("__doc__") = "pic2pic C++ code";
    def("encode_lab", ::encode_lab);
    def("encode_bgr", ::encode_bgr);
    def("ab_dict", ::ab_dict);
    def("ab_bins", ::ab_bins);
}

