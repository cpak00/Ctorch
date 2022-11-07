#include <iostream>
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;
 
int main()
{
    Mat srcImage = imread("lena.png");
    imshow("源图像",srcImage);
 
    waitKey(0);
 
    return 0;
}

