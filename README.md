# DLIP

Written by:   Si Young Lee

Course:  DLIP

Program: C++

IDE/Compiler: Visual Studio 2019

### Spatial Filter
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
