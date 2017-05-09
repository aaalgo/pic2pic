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

void stat_ab (cv::Mat *stat, cv::Mat Lab, float ab_step) {
    CHECK(Lab.type() == CV_32FC3);
    for (int i = 0; i < Lab.rows; ++i) {
        float const *row = Lab.ptr<float const>(i);
        for (int j = 0; j < Lab.cols; ++j) {
            float l = row[0];
            float a = row[1];
            float b = row[2];
            row += 3;
            CHECK(l >= 0 && l <= 100);
            CHECK(a >= AB_MIN && a <= AB_MAX);
            CHECK(b >= AB_MIN && b <= AB_MAX);

            int ia = (a - AB_MIN) / ab_step;
            int ib = (b - AB_MIN) / ab_step;
            stat->at<float>(ia, ib) += 1.0;
        }
    }
}

int main (int argc, char *argv[]) {
    float ab_step;
    {
        namespace po = boost::program_options;
        po::options_description desc_visible("Allowed options");
        desc_visible.add_options()
            ("help,h", "produce help message.")
            ("ab_step", po::value(&ab_step)->default_value(10), "")
            ;

        po::options_description desc_hidden("Allowed options");
        desc_hidden.add_options()
            ;

        po::options_description desc("Allowed options");
        desc.add(desc_visible).add(desc_hidden);

        po::positional_options_description p;
        //p.add("input", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).
                         options(desc).positional(p).run(), vm);
        po::notify(vm);


        if (vm.count("help")) {
            cout << "Usage:" << endl;
            cout << desc_visible;
            cout << endl;
            return 0;
        }
    }
    string path;

    int N = (AB_MAX - AB_MIN) / ab_step + 1;

    cv::Mat stat(N, N, CV_32F, cv::Scalar(0));  // a x b

    unsigned C = 0;
    while (cin >> path) {
        cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_COLOR);
        cv::Mat image32;
        image.convertTo(image32, CV_32FC3);
        image32 *= 1.0/255;
        cv::Mat Lab;
        cv::cvtColor(image32, Lab, CV_BGR2Lab);
        stat_ab(&stat, Lab, ab_step);
        C += Lab.total();
    }
    if (C == 0) {
        cerr << "Using random RGB image." << endl;
        cv::Mat image32(800, 800, CV_32FC3);
        cv::randu(image32, cv::Scalar(0,0,0), cv::Scalar(256, 256, 256));
        image32 *= 1.0/255;
        cv::Mat Lab;
        cv::cvtColor(image32, Lab, CV_BGR2Lab);
        stat_ab(&stat, Lab, ab_step);
    }
    cv::normalize(stat, stat, 0, 255, cv::NORM_MINMAX);
    cv::imwrite("a.png", stat);
    return 0;
}

