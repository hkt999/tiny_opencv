#include "tiny_opencv.hpp"

namespace KCV {

void inRange(const Mat &src, const Scalar lowerb, const Scalar upperb, Mat &dst) {
    if (src.empty()) {
        throw Exception("inRange: Input image is empty");
    }
    
    const int width = src.cols;
    const int height = src.rows;
    const int channels = src.channels();
    
    // Create output image (binary)
    dst.create(height, width, KCV_8UC1);
    
    uchar* dst_data = dst.getData<uchar>();
    
    if (channels == 1) {
        // Grayscale image
        const uchar* src_data = src.getData<uchar>();
        const uchar lower = (uchar)lowerb.v[0];
        const uchar upper = (uchar)upperb.v[0];
        
        for (int i = 0; i < height * width; i++) {
            uchar pixel = src_data[i];
            dst_data[i] = (pixel >= lower && pixel <= upper) ? 255 : 0;
        }
    }
    else if (channels == 3) {
        // 3-channel color image
        const cv8uc3_t* src_data = src.getData<cv8uc3_t>();
        const uchar lower_c1 = (uchar)lowerb.v[0];
        const uchar lower_c2 = (uchar)lowerb.v[1];
        const uchar lower_c3 = (uchar)lowerb.v[2];
        const uchar upper_c1 = (uchar)upperb.v[0];
        const uchar upper_c2 = (uchar)upperb.v[1];
        const uchar upper_c3 = (uchar)upperb.v[2];
        
        for (int i = 0; i < height * width; i++) {
            const cv8uc3_t& pixel = src_data[i];
            
            bool in_range = (pixel.c1 >= lower_c1 && pixel.c1 <= upper_c1) &&
                           (pixel.c2 >= lower_c2 && pixel.c2 <= upper_c2) &&
                           (pixel.c3 >= lower_c3 && pixel.c3 <= upper_c3);
            
            dst_data[i] = in_range ? 255 : 0;
        }
    }
    else {
        throw Exception("inRange: Unsupported number of channels");
    }
}

} // namespace KCV
