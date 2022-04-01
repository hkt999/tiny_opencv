#include "tiny_opencv.hpp"

namespace KCV {

void boxFilter(const Mat src, Mat &dst, int ddepth, Size ksize)
{
    // skip ddepth, force the dst color depth the same as src
    blur(src, dst, ksize);
}

} // end of namespace