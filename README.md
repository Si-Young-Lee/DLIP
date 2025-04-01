# DLIP

Written by:   Si Young Lee

Course:  DLIP

Program: C++

IDE/Compiler: Visual Studio 2019


## Spatial Filter
### Blur
```
void blur(src, dst, Kernel_Size, Anchor point);
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **Kernel_size**:  Size of Kernel Matrix

* **Anchor Point**: Reference point or reference position of an object

**Example code**
```c++
blur(src, dst, Size(i, i), Point(-1, -1));
namedWindow("Blur", WINDOW_AUTOSIZE);
imshow("Blur", dst);
```
### GausianBlur
```
void GaussianBlur(src, dst, Kernel_Size, X, Y);
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **Kernel_size**:  Size of Kernel Matrix

* **X, Y**: the standard deviation (sigma) value of the Gaussian blur filter

**Example code**
```c++
GaussianBlur(src, dst, Size(i, i), 0, 0);
namedWindow("Gaussian", WINDOW_AUTOSIZE);
imshow("Gaussian", dst);
```

### Medianblur
```
void medianBlur(src, dst, kernel_size);
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **Kernel_size**:  Size of Kernel Matrix(It should be odd)

**Example code**
```c++
medianBlur(src, dst, 3);
namedWindow("Median", WINDOW_AUTOSIZE);
imshow("Median", dst);
```

### Laplacian
```
void Laplacian(src, dst, ddepth, kernel_size, scale, delta, border_type);
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **Kernel_size**:  Size of Kernel Matrix(It should be odd)

* **scale**:  Used to adjust the result value after Laplacian operation. The default value is set to 1, and the larger the value, the stronger the edge-enhancing effect.

* **delta**:  delta is the value to be added to the Laplacian operation result. The default is set to 0, and delta is mainly used to adjust the brightness or contrast of the output.

* **border_type**: Parameters that define how pixels are processed at image boundaries.

**Example code**
```c++
int kernel_size = 3;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
Laplacian(src, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
```

### bilateral filter
```
void bilateralFilter(src, dst, d, sigmacolor, sigmaspace);
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **d**:  Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.

* **sigmacolor**:  Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.

* **sigmaspace**:  Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.

* **border_type**: Parameters that define how pixels are processed at image boundaries.

**Example code**
```c++
Mat bilateral;
bilateralFilter(src, bilateral, 3, 6, 1.5);
imshow("bilateral", bilateral);

```

### Filter2D
```
void filter2D(src, dst, ddepth, kernel_size, Anchor Point, delta, border_type);
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **Kernel_size**:  Size of Kernel Matrix(It should be odd)

* **Anchor Point**: Reference point or reference position of an object

* **delta**:  delta is the value to be added to the Laplacian operation result. The default is set to 0, and delta is mainly used to adjust the brightness or contrast of the output.

* **border_type**: Parameters that define how pixels are processed at image boundaries.

**Example code**
```c++
Mat kernel;
delta = 0;
ddepth = -1;
kernel_size = 5;
Point anchor = Point(-1, -1);
	
kernel = Mat::ones(5, 5, CV_32F);
//kernel = kernel / 25.0f; 


filter2D(src, dst, -1, kernel, anchor, 0, BORDER_DEFAULT);
```


### Threshold
```
void threshold(src, dst, thresh, maxval, type);
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **thresh**:  Value of Thresh

* **maxval**: max value of intensity

* **type**:  THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV

* **THRESH_BINARY**: Basic binarization, converting pixels larger than threshold to maxVal and smaller pixels to zero

* **THRESH_BINARY_INV**: Conversion of basic binarization, converting pixels larger than threshold to zero, and pixels smaller to maxVal

* **THRESH_TRUNC**: Hold pixels larger than threshold, and keep pixels smaller than threshold

* **THRESH_TOZERO**: Keep pixels larger than threshold and convert smaller pixels to zero

* **THRESH_TOZERO_INV**: Converts pixels larger than threshold to zero, and keeps smaller pixels intact

**Example code**
```c++

/*threshold_type
0: Binary
1: Binary Inverted
2: Threshold Truncated
3: Threshold to Zero
4: Threshold to Zero Inverted*/
int threshold_value = 130;
int threshold_type = 0;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

threshold(src, dst, threshold_value, max_BINARY_value, threshold_type);

Mat Otsu;
threshold(src, Otsu, 0, max_BINARY_value, threshold_type | THRESH_OTSU);

namedWindow("OtsuWindow", WINDOW_AUTOSIZE);
imshow("OtsuWindow", Otsu);
```

### Plot Histogram
```
calcHist(&src, Num_channel, index_channel, Mat(), b_hist, dimension, &histSize, &histRange, uniform, accumulate);
```

**Parameters**

* **src**:  input image

* **Num_channel**:  number of channel

* **index_channel**:  index of channel

* **Mat()**:  Mask image

* **b_hist**: Output of histogram(Mat)

* **dimension**: dimension of histogram(1D, 2D...)

* **&histSize**: number of intervals in the histogram

* **&histRange**: Array that sets the range of values for the histogram

* **uniform**: This value is an option to set intervals evenly when calculating histograms. When set to uniform = true, each interval is the same size. Usually set to true.

* **accumulate**: This value determines whether to accumulate histograms. Calculate a new histogram if accumulate = false; add results to the existing histogram if accumulate = true

**Example code**
```c++

//! [Compute the histograms]
Mat b_hist, g_hist, r_hist;
calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

}
```

### NORMALIZE
```
normalize(src, dst, min, max, norm_type, data_type, Mat());
```

**Parameters**

* **src**:  input array

* **dst**:  output array

* **min**:  minimum value after normalized

* **max**:  maximum value after normalized

* **norm_type**: type of normalize

* **data_type**: type of data (-1 -> no change)

* **Mat()**:  Mask image

**Example code**
```c++

int main(int argc, char** argv)
{
    //! [Load image]
    //CommandLineParser parser(argc, argv, "{@input | coin.jpg | input image}");
    //Mat src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    Mat src = imread("C:/Users/USER/source/repos/DLIP/image/coin.jpg", IMREAD_COLOR);

    if (src.empty())
    {
        return EXIT_FAILURE;
    }
    //! [Load image]

    //! [Separate the image in 3 places ( B, G and R )]
    vector<Mat> bgr_planes;
    split(src, bgr_planes);
    //! [Separate the image in 3 places ( B, G and R )]

    //! [Establish the number of bins]
    int histSize = 256;
    //! [Establish the number of bins]

    //! [Set the ranges ( for B,G,R) )]
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange = { range };
    //! [Set the ranges ( for B,G,R) )]

    //! [Set histogram param]
    bool uniform = true, accumulate = false;
    //! [Set histogram param]

    //! [Compute the histograms]
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
    //! [Compute the histograms]

    //! [Draw the histograms for B, G and R]
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    //! [Draw the histograms for B, G and R]

    //! [Normalize the result to ( 0, histImage.rows )]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    //! [Normalize the result to ( 0, histImage.rows )]

    return EXIT_SUCCESS;
}
```
### adaptiveThreshold
```
void cv::adaptiveThreshold(InputArray src, OutputArray dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C)
```

**Parameters**

* **src**:  input array

* **dst**:  Output array

* **maxValue**:  max value of intensity

* **adaptiveMethod**: ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C

* **ADAPTIVE_THRESH_MEAN_C**: Calculates the threshold based on the average value of adjacent pixels.

* **ADAPTIVE_THRESH_GAUSSIAN_C**: ADAPTIVE_THRESH_GAUSIAN_C: Calculates the threshold based on the Gaussian weighted average of adjacent pixels.

* **thresholdType**: Parameters that set the binarization method

* **THRESH_BINARY**: Allocate maxValue if it is greater than the calculated threshold, and 0 if it is less

* **blockSize**: The size of the region area for which the threshold is to be calculated. This value must be odd (for example, 3, 5, 7, 9, etc.).

* **C**: This value is subtracted from the calculated threshold, which is used to calibrate the brightness of the image.

**Example code**
```c++
//! [Compute the histograms]
void cv::adaptiveThreshold	(	
    InputArray 	src,
	OutputArray 	dst,
    double 	maxValue,
    int 	adaptiveMethod,
    int 	thresholdType,
    int 	blockSize,
    double 	C 
)	

```

### morphologyEx
```
void morphologyEx(src, dst, op, kernel, anchor, iteration, bordertype, bordervalue)
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **op**:  Type of operation(MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT)

* **kernel**: Structural elements

* **anchor**: Anchor point

* **iteration**: Number of iterations

* **bordertype**: type of border

* **borderValue**: value of border


**Example code**
```c++
Mat src = imread("image.jpg", 0);  // grayscale
Mat dst;
Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));  // Structural elements (5x5)

morphologyEx(src, dst, MORPH_OPEN, element);  //opening operation
imshow("Morphology Open", dst);
waitKey(0);	

```

### erode
```
void erode(src, dst, kernel, anchor, iteration, bordertype, bordervalue)
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **kernel**: Structural elements

* **anchor**: Anchor point

* **iteration**: Number of iterations

* **bordertype**: type of border

* **borderValue**: value of border


**Example code**
```c++
at src = imread("image.jpg", 0);  
Mat dst;
Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));  

erode(src, dst, element);  
imshow("Erosion", dst);
waitKey(0);

```

### dilate
```
void dilate(src, dst, kernel, anchor, iteration, bordertype, bordervalue)
```

**Parameters**

* **src**:  input image

* **dst**:  Output image

* **kernel**: Structural elements

* **anchor**: Anchor point

* **iteration**: Number of iterations

* **bordertype**: type of border

* **borderValue**: value of border


**Example code**
```c++
at src = imread("image.jpg", 0);  
Mat dst;
Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));  

dilate(src, dst, element);  
imshow("Erosion", dst);
waitKey(0);

```

### dilate
```
Mat getStructuringElement(int shape, Size ksize, Point anchor);
```

**Parameters**

* **shape**:  Form of structural elements (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE, etc.)
  
* **ksize**:  Size of structural elements


* **anchor**: Anchor point



**Example code**
```c++
Mat src = imread("image.jpg", 0);  // 
Mat dst;
Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));  

dilate(src, dst, element);
imshow("Dilation", dst);

erode(src, dst, element);
imshow("Erosion", dst);

morphologyEx(src, dst, MORPH_OPEN, element);
imshow("Opening", dst);

waitKey(0);
```

### findContours
```
void findContours(InputOutputArray image, OutputArrayOfArrays contours, int mode, int method,  Point offset = Point());
```

**Parameters**

* **image**:  input image
  
* **contours**:  Vector to store the found outline (in the form of std::vector<std::vector<Point>)

* **mode**: mode to find the mode outline (shape analysis method)

* **method**: Method Approximate contours

* **Point offset**: offset of point (default = (0, 0)

* **RETR_EXTERNAL**: only finds the outermost outline

* **RETR_LIST**: Find all contours without hierarchical structure

* **RETR_CCOMP**: Find all contours, hierarchically store outline and inner hole information

* **RETR_TREE**: Maintains full layer tree and finds all contours

* **CHAIN_APPROX_NONE**: Save all contour points (accurate but many data)

* **CHAIN_APPROX_SIMPLE**: Reduce the points in the straight line to save (save data)

* **CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS**: Theodor (TC89) algorithm simplifies the contour


*mode*: RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE
*method*: CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS

**Example code**
```c++
vector<vector<Point>> contours;
findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
```

### drawContours
```
void drawContours(InputOutputArray image, InputArrayOfArrays contours, int contourIdx, const Scalar& color, int thickness = 1, int lineType = 8, InputArray hierarchy = noArray(), int maxLevel = INT_MAX, Point offset = Point());
```

**Parameters**

* **image**:  input image
  
* **contours**:  coutours data (in the form of std::vector<std::vector<Point>)

* **contourIdx**: contour index to draw(all for -1)

* **color**: color of line

* **thickness**: thickness of line 

* **LineType**: Line Style (LINE_8, LINE_4, LINE_AA)

* **hierarchy**: contour hierarchy information (usually disabled)

* **maxLevel**: Specify up to what level of layer to draw (default INT_MAX)

* **Point offset**: offset of point (default = (0, 0)



**Example code**
```c++
Mat img = Mat::zeros(400, 400, CV_8UC3);
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

contours.push_back({Point(100,100), Point(200,100), Point(200,200), Point(100,200)});

drawContours(img, contours, -1, Scalar(255, 0, 0), 2);  //blue contours

imshow("Contours", img);
waitKey(0);
```

## Edge detection

### Canny
```
void Canny(InputArray image, OutputArray edges, double threshold1, double threshold2, int apertureSize=3, bool L2gradient=false )
```

**Parameters**

* **image**:  single-channel 8-bit input image.
  
* **edges**:  output edge map; it has the same size and type as image .

* **threshold1**: first threshold for the hysteresis procedure.

* **threshold2**: second threshold for the hysteresis procedure.

* **apertureSize**: aperture size for the Sobel() operator. 

* **L2gradient**: a flag, indicating whether a more accurate  L2 norm   should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L1 norm  is enough ( L2gradient=false ).Line Style (LINE_8, LINE_4, LINE_AA)


**Example code**
```c++
Mat src, src_gray;
Mat dst, detected_edges;
 
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;  // a ratio of lower:upper
int kernel_size = 3; //Sobel Operation
String window_name = "Edge Map";

Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

```

### HoughLines
```
void HoughLines(InputArray image, OutputArray lines, double rho, double theta, int threshold, double srn=0, double stn=0 )
```

**Parameters**

* **image**:  Input image
  
* **lines**:  Output lines

* **rho, theta**: HoughLines return polar coordinates (rho, theta)

* **threshold**: Thresholding value

* **srn**: If the default value is 0, we use standard Hough transform, and if it is greater than 0, we use multi-scale Hough transform. If srn > 0, the rho value is further subdivided to detect the straight line.

* **stn**: If the default value is 0, use standard Hough transform, and if it is greater than 0, use multiscale Hough transform. If stn > 0, theta values can be further subdivided to detect more candidate straight lines


**Example code**
```c++

// (Option 1) Standard Hough Line Transform
vector<Vec2f> lines;
HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

// Draw the detected lines
for (size_t i = 0; i < lines.size(); i++)
{
	float rho = lines[i][0], theta = lines[i][1];
	Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a * rho, y0 = b * rho;
	pt1.x = cvRound(x0 + 1000 * (-b));
	pt1.y = cvRound(y0 + 1000 * (a));
	pt2.x = cvRound(x0 - 1000 * (-b));
	pt2.y = cvRound(y0 - 1000 * (a));
	line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
}

```

### HoughLinesP
```
void HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )

**Parameters**

* **image**:  Input image
  
* **lineP**:  Output lines

* **rho, theta**: HoughLines return polar coordinates (rho, theta)

* **threshold**: Thresholding value

* **minLineLength**: Minimum straight length to be detected (shorter lines are ignored)

* **maxLineGap**: Maximum allowable spacing between line fragments (greater than this, considered another line)


**Example code**
```c++


// (Option 2) Probabilistic Line Transform
vector<Vec4i> linesP;
HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10);

// Draw the lines
for (size_t i = 0; i < linesP.size(); i++)
{
	Vec4i l = linesP[i];
	line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
}

```

### HoughCircles
```
void HoughCircles(InputArray image, OutputArray circles, int method, double dp, double minDist, double param1 = 100, double param2 = 100, int minRadius = 0, int maxRadius = 0);)

**Parameters**

* **image**:  Input image
  
* **circles**:  Output circles

* **method**: Huff conversion method (currently HOUGH_GRADIENT only available)

* **dp**: Resolution Ratio (usually 1.0 or 1.5 used)

* **minDist**: Minimum distance between detected circle centers

* **param1**: High threshold for Canny Edge detector

* **param2**: Threshold of Huff Transformation for Circle Detection

* **minRadius**: Minimum circle radius to detect

* **maxRadius**: Maximum circle radius to detect


**Example code**
```c++
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat src, gray;
	
	String filename = "pillsetc.png";
	
	/* Read the image */
	src = imread(filename, 1);
	
	if (!src.data)
	{
		printf(" Error opening image\n");
		return -1;
	}
		
	cvtColor(src, gray, COLOR_BGR2GRAY);

	/* smooth it, otherwise a lot of false circles may be detected */
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);

	vector<Vec3f> circles;
	HoughCircles(gray, circles, 3, 2, gray.rows / 4, 200, 100);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		/* draw the circle center */
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);

		/* draw the circle outline */
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}

	namedWindow("circles", 1);
	imshow("circles", src);
	
	/* Wait and Exit */
	waitKey();
	return 0;
}

```
