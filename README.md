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

