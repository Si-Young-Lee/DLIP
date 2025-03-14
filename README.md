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

* **type**:  THRESH_BINARY: Basic binarization, converting pixels larger than threshold to maxVal and smaller pixels to zero
THRESH_BINARY_INV: Conversion of basic binarization, converting pixels larger than threshold to zero, and pixels smaller to maxVal
THRESH_TRUNC: Fix pixels larger than threshold and keep pixels smaller than threshold
THRESH_TOZERO: Keep pixels larger than threshold, convert smaller pixels to zero
THRESH_TOZERO_INV: Converts pixels larger than threshold to zero, keeps smaller pixels intact

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

    //! [Draw for each channel]
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }
    //! [Draw for each channel]

    //! [Display]
    imshow("Source image", src);
    imshow("calcHist Demo", histImage);
    waitKey();
    //! [Display]

    return EXIT_SUCCESS;
}
```
