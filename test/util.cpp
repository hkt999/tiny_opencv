#include <stdlib.h>
#include <string.h>
#include "tiny_opencv.hpp"
#include "unit_test.hpp"

using namespace KCV;

void *dup_mat_data( Mat &mat )
{
    void *mem = 0;
    int size = mat.cols * mat.rows * mat.elemSize();
    mem = malloc(size);
    memcpy(mem, mat.ref->data, size);

    return mem;
}