#include <stdlib.h>
#include <string.h>
#include "tiny_opencv.hpp"
#include <assert.h>

namespace KCV {

Mat::Mat():data(0), rows(0), cols(0), type(0), ref(0)
{
}

void Mat::createBuffer()
{
    if (ref) {
        if (--ref->count <=0 ) {
            free(ref->data);
            free(ref);
            ref = 0;
        }
    }

    ref = (DataRef *)malloc(sizeof(DataRef));
    ref->count++;
    data = ref->data = malloc( cols * rows * elemSize());
}

void Mat::create(int _rows, int _cols, int _type)
{
    rows = _rows;
    cols = _cols;
    type = _type;

    createBuffer();
}

void Mat::release()
{
    if (ref) {
        if (--ref->count <= 0) {
            free(ref->data);
            free(ref);
            ref = 0;
        }
    }
}

Mat::Mat(int rows, int cols, int type): rows(rows), cols(cols), type(type), ref(0)
{
    createBuffer();
}

Mat::Mat(Size size, int type): rows(size.height), cols(size.width), type(type), ref(0)
{
    createBuffer();
}

Mat::Mat(const Mat &m): rows(m.rows), cols(m.cols), type(m.type), data(m.data), ref(m.ref)
{
}

Mat::Mat(int rows, int cols, int type, void *data): rows(rows), cols(cols), type(type), data((uchar *)data), ref(0)
{
}

Mat::Mat(Size size, int type, void *data): rows(size.height), cols(size.width), type(type), data((uchar *)data), ref(0)
{
}

template <typename _Tp> static void copyBlock(Mat &mdst, const Mat &msrc, const Rect2i &roi)
{
    _Tp *start = msrc.getData<_Tp>(roi.y, roi.x);
    _Tp *dst = mdst.getData<_Tp>();
    int row_count = roi.height;
    while (row_count-->0) {
        memcpy(dst, start, roi.width * sizeof(_Tp));
        start += msrc.cols;
        dst += mdst.cols; 
    }
}

Mat::Mat(const Mat &m, const Rect &roi):cols(roi.width), rows(roi.height), type(m.type), ref(0)
{
    createBuffer();
    switch (type) {
        case CV_8UC1: 
            copyBlock<uchar>(*this, m, roi);
            break;

        case CV_8UC3:
            copyBlock<cv8uc3_t>(*this, m, roi); 
            break;

        case CV_32F:
            copyBlock<float>(*this, m, roi);
            break;

        case CV_64F:
            copyBlock<double>(*this, m, roi);
            break;

        default:
            throw Exception("mat constructor (m/roi) not supporting types");
    }
}

// destructor
Mat::~Mat()
{
    if (ref) {
        if (--ref->count <= 0) {
            free(ref->data);
            free(ref);
            ref = 0;
        }
    }
}

// operator
Mat& Mat::operator=(const Mat& m)
{
    rows = m.rows;
    cols = m.cols;
    type = m.type;
    data = m.data;
    ref = m.ref;
    if (ref) {
        ref->count++;
    }

    return *this;
}

Mat& Mat::operator+(const Mat &m) const
{
    if ((cols != m.cols) || (rows != m.rows))
        throw Exception("operator + dimention is mismatched");

    Mat *t = new Mat(rows, cols, type);
    t->createBuffer();

    if (type == CV_32F) {
        float *dst = t->getData<float>(), *a = getData<float>(), *b = m.getData<float>();
        int count = rows * cols;
        while (count-->0) {
            *dst++ = *a++ + *b++;
        }
    }

    return *t;
}

Mat &Mat::operator+=(const Mat &m)
{
    if ((cols != m.cols) || (rows != m.rows))
        throw Exception("operator + dimention is mismatched");

    if (type == CV_32F) {
        float *dst = getData<float>(), *a = getData<float>(), *b = m.getData<float>();
        int count = rows * cols;
        while (count-->0) {
            *dst++ = *a++ + *b++;
        }
    }

    return *this;
}

Mat& Mat::operator-(const Mat &m) const
{
    if ((cols != m.cols) || (rows != m.rows))
        throw Exception("operator + dimention is mismatched");

    if (type != m.type)
        throw Exception("type mismatch for matrix type");

    Mat *t = new Mat(rows, cols, type);
    t->createBuffer();

    if (type == CV_32F) {
        float *dst = t->getData<float>(), *a = getData<float>(), *b = m.getData<float>();
        int count = rows * cols;
        while (count-->0) {
            *dst++ = *a++ - *b++;
        }
    }

    return *t;
}

Mat& Mat::operator*(const Mat& m) const
{
    assert(cols == m.rows);
    assert(type == m.type);

    Mat &t = Mat::zeros(rows, m.cols, type);
    float *c = t.getData<float>();
    float *la = getData<float>();
    for (int i=0; i<rows; i++) {
        float *lb = m.getData<float>();
        for (int j=0; j<m.cols; j++) {
            float *a = la, *b = lb;
            float elemValue = 0;
            for (int k=0; k<cols; k++) {
                elemValue += *a++ * *b;
                b += m.cols;
            }
            *c++ = elemValue;
            lb++;
        }
        la += cols;
    }
    return t; 
}

Mat& Mat::clone()
{
    Mat *m = new Mat(rows, cols, type);
    m->createBuffer();

    return *m;
}

int Mat::elemSize()
{
    static int mapping[] = { 1, 1, 2, 2, 4, 4, 8, 0};
    return channels() * mapping[depth()];
}

Mat &Mat::operator()( const Rect& roi )
{
    Mat *t = new Mat(roi.height, roi.width, type);

    int src_line_bytes = cols * elemSize();
    int dst_line_bytes = t->cols * t->elemSize();
    int h = roi.height;
    uchar *src = (uchar *)data + (roi.y * cols + roi.x) * elemSize();
    uchar *dst = (uchar *)t->data;
    while (h-->0) {
        memcpy(dst, src, dst_line_bytes);
        src += src_line_bytes;
        dst += dst_line_bytes;
    }

    return *t;
}

template <typename _Tp> static void do_transpose(Mat &mdst, Mat &msrc)
{
    for (int i=0; i<msrc.rows; i++) {
        for (int j=0; j<msrc.cols; j++) {
            mdst.at<_Tp>(j,i) = msrc.at<_Tp>(i,j);
        }
    }
}

Mat& Mat::transpose()
{
    Mat *t = 0;
    switch (type) {
        case CV_8U:
            t = new Mat( cols, rows, CV_8U);
            do_transpose<uchar>(*t, *this);
            break;
        case CV_32F:
            t = new Mat( cols, rows, CV_32F);
            do_transpose<float>(*t, *this);
            break;
        case CV_64F:
            t = new Mat( cols, rows, CV_64F);
            do_transpose<double>(*t, *this);
            break;
        default:
            throw Exception("not supporting data type for matrix transpose");
    }

    return *t;
}

float Mat::determinant()
{
    float det = 0;

    if (type != CV_32FC1)
        return 0; // TODO: throw exception

    switch (rows) {
        case 1: {
            return *getData<float>();
        }
        break;
        case 2: {
            float *src = getData<float>();
            det = src[0] * src[3] - src[1] * src[2];
            return det;
        }
        break;
        case 3: {
            float *src = getData<float>();
            float a = src[0];
            float b = src[1];
            float c = src[2];
            float d = src[3];
            float e = src[4];
            float f = src[5];
            float g = src[6];
            float h = src[7];
            float i = src[8];

            return (a*e*i + b*f*g + c*d*h) - (a*f*h + b*d*i + c*e*g);
        }
        break;

        default: {
            int DIM = rows;
            Mat temp(DIM-1, DIM-1, type);
            float det = 0;
            float *det_src = getData<float>();
            float *src_start_line = det_src + cols;
            for(int k = 0; k < DIM; k++) {
                float *src = src_start_line;
                float *dst = (float *)temp.data;
                for(int i = 1; i < DIM; i++) {
                    for(int j = 0; j < DIM; j++) {
                        if(k == j) {
                            src++;
                            continue;
                        }
                        *dst++ = *src++;
                    }
                } 
                if( (k&1) == 0) {
                    det += *det_src++ * temp.determinant();
                } else {
                    det -= *det_src++ * temp.determinant();
                }
            }
            return det;
        }
        break;
    }
}

Mat &Mat::cofactor_()
{
    Mat *cofactor = new Mat(rows, cols, type);
    if (rows != cols)
        return *cofactor;

    if (rows == 1) {
        float *dst = cofactor->getData<float>();
        float *src = getData<float>();
        *dst = *src;
        return *cofactor;
    } else if (rows == 2) {
        float *src = getData<float>() + 3;
        float *dst = cofactor->getData<float>();
        *dst++ = *src--;
        *dst++ = -(*src--);
        *dst++ = -(*src--);
        *dst++ = *src;
        return *cofactor;
    } 

    Mat temp(rows-1, rows-1, type);
    bool flagPositive = true;
    float *co = cofactor->getData<float>();
    for(int k1 = 0; k1 < rows; k1++) {  
        flagPositive = ( (k1 & 1) == 0);
        for(int k2 = 0; k2 < rows; k2++) {
            float *src = getData<float>();
            float *dst = temp.getData<float>();
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < rows; j++) {
                    if(k1 == i || k2 == j) {
                        src++;
                        continue;
                    }
                    *dst++ = *src++;
                }
            }

            float det = temp.determinant();
            if (flagPositive) {
                *co++ = det;
                flagPositive = false;
            } else {
                *co++ = -det;
                flagPositive = true;
            }
        }
    }

    return *cofactor;
}

Mat& Mat::inverse()
{
    Mat *inv = new Mat(rows, cols, type);
    if (rows != cols)
        throw Exception("mismatch rows/cols in inverse");

    // 1x1
    if (rows == 1) {
        inv->at<float>(0,0) = 1 / at<float>(0,0);
        return *inv;
    }

    // to find out determinant
    float det = determinant();
	Mat &cofactor = cofactor_();
    // inv = transpose of cofactor / Determinant
    float *src = cofactor.getData<float>();
    float *dst_line = inv->getData<float>();
    for(int i = 0; i < rows; i++) {
        float *dst = dst_line;
        for(int j = 0; j < cols; j++) {
            *dst = *src++ / det;
            dst += cols;
        }
        dst_line++;
    }

    return *inv;
}

Mat& Mat::zeros(int cols, int rows, int type)
{
    Mat *t = new Mat(cols, rows, type);
    t->createBuffer();
    bzero(t->data, cols * rows * t->elemSize());

    return *t;
}

template <typename _Tp> inline void onesRow(Mat &dst)
{
    for (int i=0; i<dst.rows; i++) {
        dst.at<_Tp>(i, 0) = 1;
    }
}

Mat& Mat::ones(int cols, int rows, int type)
{
    Mat &t = zeros(cols, rows, type);
    switch (t.depth()) {
        case CV_8U:
            onesRow<uchar>(t);
            break;

        case CV_32F:
            onesRow<float>(t);
            break;

        default:
            throw Exception("ones doesn't support this type");
    }

    return t;
}

Mat& Mat::eye(int i_rows, int i_cols, int type)
{
    Mat &t = zeros(i_rows, i_cols, type);
    float *d = t.getData<float>();
    int count = MIN(i_rows, i_cols);
    while (count-->0) {
        *d = 1;
        d = d + t.cols + 1;
    }
    return t;
}

void Mat::copyTo(Mat &dest)
{
    dest.release();
    dest.create(rows, cols, type);
    if (ref->data) {
        memcpy(dest.ref->data, ref->data, cols * rows * elemSize());
    }
}

void setIdentity(Mat &mat) 
{
    assert(mat.ref != 0);
    assert(mat.ref->data != 0);

    memset(mat.ref->data, 0, mat.cols * mat.rows * mat.elemSize());
    int loop = mat.rows > mat.cols ? mat.cols : mat.rows;
    for (int i=0; i<loop; i++) {
        void *t = (uint8_t *) mat.ref->data + (i * mat.cols + i) * mat.elemSize();
        switch (mat.type) {
        case CV_8U: 
            *((int8_t *)t) = 1;
            break;

        case CV_32F:
            *((float *)t) = 1.0;
            break;

        case CV_64F:
            *((double *)t) = 1.0;
            break;
        }
    }
}

template <typename T> static void pack_data(void *dest, double *src, int count)
{
    T *d = (T *)dest;
    for (int i=0; i<count; i++) {
        *d++ = (T)(*src++);
    }
}

template <typename T> static void copy_data(void *dest, void *src) 
{
    *((T *)dest) = *((T *)src);
}


typedef void (*copy_data_f)(void *dest, void *src);
typedef void (*pack_data_f)(void *dest, double *src, int count);

void setIdentity(Mat &mat, Scalar sv)
{
    uint8_t data[8*8];
    int numCol = (mat.type >> 3) + 1;
    assert(numCol <= 4);

    int d_type = mat.type & 0x7;

    assert(mat.ref != 0);
    assert(mat.ref->data != 0);

    // clear the matrix reset to 0
    memset(mat.ref->data, 0, mat.cols * mat.rows * mat.elemSize());
    static copy_data_f c[] = {
        copy_data<uint8_t>,
        copy_data<int8_t>,
        copy_data<uint16_t>,
        copy_data<int16_t>,
        copy_data<int32_t>,
        copy_data<float>,
        copy_data<double>,
        copy_data<double>
    };

    static pack_data_f p[] = {
        pack_data<uint8_t>,
        pack_data<int8_t>,
        pack_data<uint16_t>,
        pack_data<int16_t>,
        pack_data<int32_t>,
        pack_data<float>,
        pack_data<double>,
        pack_data<double>
    };

    pack_data_f pf = p[d_type];
    pf(data, sv.v, sv.count);

    copy_data_f cf = c[d_type];

    int loop = mat.rows > mat.cols ? mat.cols : mat.rows;
    for (int i=0; i<loop; i++) {
        void *t = (uint8_t *) mat.ref->data + (i * mat.cols + i) * mat.elemSize();
    }
}

#include <stdio.h>
void Mat::print(const char *name)
{
    printf("matrix name: %s\n", name);
    printf("    rows=%d, cols=%d\n", rows, cols);
}

} // end of namespace (KCV)
