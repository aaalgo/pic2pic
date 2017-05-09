#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include "colorize.h"

// Lab: L in [0, 100]
//      a in [-127, 127]
//      b in [-127, 127]

using namespace std;

int main (int argc, char *argv[]) {
    cv::Mat lenna = cv::imread("lenna.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat lenna32;
    lenna.convertTo(lenna32, CV_32FC3);
    lenna32 *= 1.0/255;
    cv::Mat Lab;
    cv::cvtColor(lenna32, Lab, CV_BGR2Lab);
    AB codec;
    cv::Mat code(lenna.rows * lenna.cols, AB::BINS, CV_32F, cv::Scalar(0));
    AB::dists_buffer_t dists;

    for (unsigned xxx = 0; xxx < 30; ++xxx) {
        int cc = 0;
        for (int i = 0; i < Lab.rows; ++i) {
            float const *row = Lab.ptr<float const>(i);
            for (int j = 0; j < Lab.cols; ++j) {
                float *bins = code.ptr<float>(cc);
                float w;
                codec.encode(row + 1, bins, &w, dists);
                row += 3;
                if (cc < 20 && xxx == 0)
                for (unsigned k = 0; k < AB::BINS; ++k) {
                    if (bins[k] != 0) {
                        cout << '(' << (i * Lab.cols + j) << ", " << k << ")\t" << bins[k] << endl;
                    }
                }
                ++cc;
            }
        }
        cout << '.' << endl;
    }
    return 0;
}

