#ifndef AAALGO_COLORIZE
#define AAALGO_COLORIZE 1

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

static float constexpr AB_MIN = -127.0;
static float constexpr AB_MAX = 127.0;

class AB {
    // lookup table to speed up encoding
    cv::Mat lookup;
    int a_bin_min;
    int b_bin_min;
    std::vector<float> W;
public:
    static float constexpr BIN_STEP = 10.0;
    static int constexpr BINS = 313;    //
    static int constexpr SEARCH_RADIUS = 2;
    static int constexpr SEARCH_W = 2 * SEARCH_RADIUS + 1;
    static int constexpr DISTS_BUFFER = SEARCH_W * SEARCH_W;
    typedef std::pair<float, int> dists_buffer_t[DISTS_BUFFER];
    static int constexpr K = 10;        // K-NN soft assignment
    static float constexpr sigma = 5.0;
    static float constexpr sigma_2sqr = 2 * sigma * sigma;
    static float constexpr gamma = 0.5;
    static float constexpr alpha = 1.0;
    static float const PRIOR[];
    static float const CC[][2];
    // bins must be initialized to 0s
    AB ();
    void encode (float const *ab, float *bins, float *w, std::pair<float, int> *);
};

#endif
