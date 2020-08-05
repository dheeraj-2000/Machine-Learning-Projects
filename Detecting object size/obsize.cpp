#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<stdio.h>
using namespace std;
using namespace cv;

int main()

{

Mat f=imread("/home/dheeraj/my_projects/my_project_env/practice/motion_detector/Detecting object size/images.jpeg");//positive
Mat hsv_th;
cvtColor(f,f,CV_BGR2HSV);

inRange(f,Scalar(0,100,0),Scalar(100,255,100),hsv_th);
dilate(hsv_th,hsv_th,cv::Mat());
dilate(hsv_th,hsv_th,cv::Mat());
dilate(hsv_th,hsv_th,cv::Mat());
dilate(hsv_th,hsv_th,cv::Mat());


for(;;)
{
  imshow("fore",f);
  imshow("hsv",hsv_th);

 char c=waitKey(10);
 if(c=='b')//break when 'b' is pressed
      {
         break;
      }
 }
return 0;
}
