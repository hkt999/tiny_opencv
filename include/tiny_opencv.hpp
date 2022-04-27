#ifndef _OPENCV_CORE_H_
#define _OPENCV_CORE_H_

#include "tiny_types.hpp"

#define KCV kcv

namespace KCV {

class Exception {
	public:
		const char *message;

	public:
		Exception(const char *message):message(message) {}
		~Exception() {}

		inline const char *what() const { return message; }
};

template <typename _Tp> class Size_
{
	public:
		typedef _Tp value_type;
		_Tp width;
		_Tp height;

	public:
		Size_() {}
		Size_(_Tp _width, _Tp _height): width(_width), height(_height) {}
		Size_(const Size_<_Tp>& sz): width(sz.width), height(sz.height) {}
		Size_& operator = (const Size_<_Tp>& sz) {
			width = sz.width;
			height = sz.height;

			return *this;
		}
		_Tp area() const {
			return width * height;
		}
		double aspectRatio() const {
			return (double)width / (double)height;
		}
		bool empty() const {
			return ((width==0) || (height==0)) ? true : false;
		}

		template<typename _Tp2> operator Size_<_Tp2>() const;
};

typedef Size_<int> Size2i;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;

template <typename _Tp> class Rect_;
template <typename _Tp> class Point_
{
	public:
		typedef _Tp value_type;
		_Tp x;
		_Tp y;

	public:
		Point_() {}
		Point_(_Tp _x, _Tp _y):x(_x), y(_y) {}
		Point_(const Point_<_Tp>& pt):x(pt.x), y(pt.y) {}
		Point_<_Tp>& operator = (const Point_<_Tp>& pt) {
			x = pt.x;
			y = pt.y;
			return *this;
		}
		bool inside(const Rect_<_Tp>& r) const {
			return ((x>=r.x)&&(y>=r.y)&&(x<=r.x + r.width-1)&&(y<=r.y + r.height-1)) ? true : false;
		}

};

typedef Point_<int> Point2i;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;

#define MIN(a,b)    (((a)<(b))?(a):(b))
#define MAX(a,b)    (((a)<(b))?(b):(a))
template <typename _Tp> class Rect_
{
	public:
		typedef _Tp value_type;

		_Tp x;
		_Tp y;
		_Tp width;
		_Tp height;

	public:
		Rect_() { }
		Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height): x(_x), y(_y), width(_width), height(_height) {}
		Rect_(const Rect_<_Tp>& r): x(r.x), y(r.y), width(r.width), height(r.height) {}
		Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz): x(org.x), y(org.y), width(sz.width), height(sz.height) {}
		Rect_<_Tp>& operator = ( const Rect_<_Tp>& r ) {
			x = r.x;
			y = r.y;
			width = r.width;
			height = r.height;
			return *this;
		}
		Point_<_Tp> tl() const {
		    return Point_<_Tp>(x,y);
		}
		Point_<_Tp> br() const {
			return Point_<_Tp>(x + width - 1, y + height - 1);
		}
		Size_<_Tp> size() const {
			return Size_<_Tp>(width, height);
		}
		_Tp area() const {
			return width * height;
		}
		bool empty() const {
			return ((width==0) || (height==0)) ? true : false;
		}
		bool contains(const Point_<_Tp>& pt) const {
			return ((pt.x>=x)&&(pt.y>=y)&&(pt.x<=x+width-1)&&(pt.y<=y+height-1)) ? true : false;
		}
};

typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect_<double> Rect2d;
typedef Rect2i Rect;

// +--------+----+----+----+----+------+------+------+------+
// |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
// +--------+----+----+----+----+------+------+------+------+
// | CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
// | CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
// | CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
// | CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
// | CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
// | CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
// | CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
// +--------+----+----+----+----+------+------+------+------+

enum {
	CV_8UC1 =  0, CV_8SC1, CV_16UC1, CV_16SC1, CV_32SC1, CV_32FC1, CV_64FC1,
	CV_8UC2 =  8, CV_8SC2, CV_16UC2, CV_16SC2, CV_32SC2, CV_32FC2, CV_64FC2,
	CV_8UC3 = 16, CV_8SC3, CV_16UC3, CV_16SC3, CV_32SC3, CV_32FC3, CV_64FC3,
	CV_8UC4 = 24, CV_8SC4, CV_16UC4, CV_16SC4, CV_32SC4, CV_32FC4, CV_64FC4
};

enum {
	CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3, CV_32S=4, CV_32F=5, CV_64F=6
};

#pragma pack(push, 1)
typedef struct _8uc1_t {
    uchar c1;
} cv8uc1_t;

typedef struct _8uc3_t {
    uchar c1;
    uchar c2;
    uchar c3;
} cv8uc3_t;
#pragma pack(pop)

typedef struct _DataRef {
	void *data;
	int count;
} DataRef;

class Scalar {
	public:
		int count;
		double v[4];

	public:
		Scalar():Scalar(0) {}
		Scalar(double v0) { count = 1; v[0] = v0; v[1] = 0; v[2] = 0; v[3] = 0; }
		Scalar(double v0, double v1) { count = 2; v[0] = v0; v[1] = v1; v[2] = 0; v[3] = 0; }
		Scalar(double v0, double v1, double v2) { count = 3; v[0] = v0; v[1] = v1; v[2] = v2; v[3] = 0; }
		Scalar(double v0, double v1, double v2, double v3) { count=4; v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3; }
		static Scalar all(double v) { Scalar s(v,v,v,v); return s; }
};

class Mat
{
	public:
		// public members
		int rows;
		int cols;
		int type;
		DataRef *ref;

	public:
		// constructor
		Mat();
		Mat(int rows, int cols, int type);
		Mat(Size size, int type);
		Mat(const Mat &m);
		Mat(int rows, int cols, int type, void *data);
		Mat(Size size, int type, void *data);
		Mat(const Mat &m, const Rect2i &roi);

		virtual ~Mat();	

	public:
		// operator
		Mat& operator=(const Mat& m);
		Mat& operator+=(const Mat &m);
		Mat& operator+(const Mat& m) const;
		Mat& operator-(const Mat& m) const;
		Mat& operator*(const Mat& m) const;
		Mat& clone();

		void createBuffer();
		void create(int rows, int cols, int type);
		template <typename _Tp> inline _Tp *getData() const {
			return (_Tp *)ref->data;
		}
		template <typename _Tp> inline _Tp *getData(int i_row, int i_col) const {
			_Tp *d = (_Tp *)ref->data;
			return d + (cols * i_row + i_col);
		}
		template <typename _Tp> inline _Tp& at(int i_row, int i_col) const {
			_Tp *d = (_Tp *)ref->data;
		    return d[cols * i_row + i_col];
		}
		template <typename _Tp> inline _Tp& at(int idx) const {
			_Tp *d = (_Tp *)ref->data;
			return d[idx];
		}
		void *ptr() const {
			return (void *)ref->data;
		}
		Mat& transpose();
		inline Mat &t() { return transpose(); }

		float determinant();
		Mat& cofactor_();
		Mat& inverse();

		Mat &operator()( const Rect& roi );
		static Mat& zeros(int rows, int cols, int type);
		static Mat& ones(int rows, int cols, int type);
		static Mat& eye(int rows, int cols, int type);

	public:
		// member functions
		const inline bool empty() { return (ref->data == 0) || ((cols == 0) && (rows==0)); }
		inline int channels() { return (type/8)+1; }
		inline int depth() { return (type%8); }
		inline int dims() { return 2; } // only supports 2 channels (row/col)
		int elemSize();
		void release();
		void copyTo(Mat &dest);
		void print(const char *name); // for debug
};

void setIdentity(Mat &mat);
void setIdentity(Mat &mat, Scalar value);

// Color transformation
enum {
	CV_BGR2GRAY = 0,
	CV_RGB2GRAY,
	CV_GRAY2BGR,
	CV_GRAY2RGB,
	CV_BGR2YUV_I420,
	CV_RGB2YUV_I420,
	CV_YUV2BGR_I420,
	CV_YUV2RGB_I420,
	CV_RGB2BGR,
	CV_BGR2RGB,
	CV_BGR2HSV,
	CV_RGB2HSV,
	CV_HSV2BGR,
	CV_HSV2RGB
};

// OpenCV matrix(image) functions
void cvtColor(const Mat src, Mat &dst , int code);

enum {
	INTER_NEAREST = 0,
	INTER_LINEAR = 1,
	INTER_CUBIC = 2
};
void resize(const Mat src, Mat &dst, Size size, float h_ratio = 0.0, float v_ratio = 0.0, int mode = INTER_NEAREST);

enum {
	THRESH_BINARY = 0,
	THRESH_BINARY_INV,
	THRESH_TRUNC,
	THRESH_TOZERO,
	THRESH_TOZERO_INV
};

// algorithms
void threshold(const Mat in, Mat &out, double thresh, double maxval, int type);
void blur(const Mat src, Mat &dst, Size ksize);
void bilateralFilter(const Mat src, Mat &dst, int d, double sigmaColor, double sigmaSpace);
void boxFilter(const Mat src, Mat &dst, int ddepth, Size ksize);
void gaussianBlur(const Mat src, Mat &dst, Size ksize, double sigmaX, double sigmaY=0.0);
void medianBlur(const Mat src, Mat &dst, int ksize);
void filter2D(const Mat src, Mat &dst, int ddepth, Mat &kernel);
void equalizeHist(const Mat src, Mat &dst);
Mat getRotationMatrix2D(Point2f center, double angle, double scale);
Mat getAffineTransform(const Point2f src[], const Point2f dst[]);

// utilities
void randn(Mat &dst, float mean, float sigma);

}; // end of namespace

#endif
