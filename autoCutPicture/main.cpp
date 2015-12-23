//
//  main.cpp
//  autoCutPicture
//
//  Created by rondou.chen on 2014/10/18.
//  Copyright (c) 2014年 rondou chen. All rights reserved.
//

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdlib.h>
#include "cvaux.h"
#include "highgui.h"
#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;

int getMaskSize(int isXorY, int listLength, Mat frame);

Mat origin_image;
Mat origin_image_line;
Mat final_origin_image;
Mat src_gray;
Mat src_ycbcr;
Mat src_lab;
Mat canny_output;
Mat integral_output;
Mat integral_output1;

int canny_output_pix_size;
int finalX;
int finalY;
int finalSizeX;
int finalSizeY;
int MaskSizeX = 0;
int MaskSizeY = 0;
int temp = 0;

//Mat test_gary;


/// Generate grad_x and grad_y
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
Mat grad;

//string origin_file("/Users/rondouchen/Pictures/summerwar.jpg");
//string origin_file("/Users/rondouchen/Pictures/summergirl.jpg");
//string origin_file("/Users/rondouchen/Pictures/testjump.jpg");
//string origin_file("/Users/rondouchen/Pictures/testjump2.jpg");
//string origin_file("/Users/rondouchen/Pictures/test000.jpg");
string origin_file("/Users/rondouchen/Pictures/rain.jpg");
//string origin_file("/Users/rondouchen/Pictures/rain2.jpg");
//string origin_file("/Users/rondouchen/Pictures/test.jpg");

//string origin_file("/Users/rondouchen/Pictures/201309031.png");



int main(int argc, const char * argv[])
{
    // insert code here...
    origin_image = imread(origin_file, 1); // resoure image
    cv::imshow("show_origin", origin_image);
    cvtColor(origin_image, src_gray, CV_BGR2GRAY); /// 轉灰
    //cvtColor(origin_image, src_ycbcr, CV_RGB2YCrCb, 0); /// TO YcBcR
    cvtColor(origin_image, src_ycbcr, CV_RGB2YCrCb); /// TO YcBcR
    cvtColor(origin_image, src_lab, CV_RGB2Lab);
    
    //cvtColor(origin_image, test_gary, CV_BGR2GRAY); /// 轉灰
    //integral(test_gary, test_gary);
    
    Mat yyy(src_ycbcr.rows, src_ycbcr.cols, DataType<uchar>::type);
    Mat lll(src_lab.rows, src_lab.cols, DataType<uchar>::type);
    //cout << "dcccccc = " << 5/0;
    
    for (int i = 0; i < src_ycbcr.rows; ++i) {
        for (int j = 0; j < src_ycbcr.cols; ++j) {
            cv::Vec3f pixel = src_ycbcr.at<cv::Vec3b>(i, j);
            if (src_ycbcr.at<Vec3b>(i, j)[0] != 0) {
                yyy.at<uchar>(i, j) = (((((pixel[1] + pixel[2]))/2 / pixel[0])));
                //cout << ((((pixel[1] + pixel[2])) / pixel[0])) << endl;
            } else {
                yyy.at<uchar>(i, j) = ((((pixel[1] + pixel[2])))) - ((((pixel[1] + pixel[2]))));
                //cout << ((((pixel[1] + pixel[2])))) << endl;
            }
            //cout << (((pixel[1] + pixel[2]) / 2) / pixel[0]) << endl;
        }
    }
    

    for (int i = 0; i < src_lab.rows; i++) {
        for (int j =0; j < src_lab.cols; j++) {
            //cv::Vec3f pixel = src_lab.at<cv::Vec3b>(i, j);
            if (src_lab.at<Vec3b>(i, j)[0] != 0) {
                //lll.at<uchar>(i, j) = (src_lab.at<Vec3b>(i, j)[1] + src_lab.at<Vec3b>(i, j)[2]) / 2;
                lll.at<uchar>(i, j) = (src_lab.at<Vec3b>(i, j)[1] + src_lab.at<Vec3b>(i, j)[2]) / 2 / src_lab.at<Vec3b>(i, j)[0];
            } else {
                //cout <<"xxxx = "<< (src_lab.at<Vec3b>(i, j)[1] + src_lab.at<Vec3b>(i, j)[2]) / 2 << endl;
                lll.at<uchar>(i, j) = (src_lab.at<Vec3b>(i, j)[1] + src_lab.at<Vec3b>(i, j)[2]) / 2;
                //lll.at<uchar>(i, j) = (uchar)0;
            }
            //lll.at<uchar>(i, j) = (((pixel[1] + pixel[2]) / 2) / (pixel[0]));
        }
    }
    
    /*for (int i = 0; i < yyy.rows; i++) {
        for (int j =0; j < yyy.cols; j++) {
            cout << "dddd" << endl;
            cout << (double)yyy.at<uchar>(i, j) << endl;
        }
    }*/
    
    //Mat gray_mat = src_gray;
    //cv::imshow("show cY", yyy);
    Mat gray_mat = yyy;//yyy or lll
    const unsigned short FILTER_RADIUS = 13;
    // cout << "gray_mat = " << gray_mat << endl;
    // 遍歷整張圖的 pixel
    
    // medianBlur(gray_mat, gray_mat, 5);//模糊化
    Mat xxx(gray_mat.rows, gray_mat.cols, DataType<uchar>::type);
    
    for (int i = 0; i < gray_mat.rows; ++i) {
        for (int j = 0; j < gray_mat.cols; ++j) {
            // 原始中點值
            // int center_value = static_cast<int>(gray_mat.at<uchar>(i, j));
            int center_value = gray_mat.at<uchar>(i, j);
            //cout << "center_value = " << center_value << endl;
            // 處歷過的中點值
            int processed_center_value = 0;
            // 一圈一圈往外繞來過濾
            int L = 0;
            int A = 0;
            int B = 0;
            for (int r = 1; r <= FILTER_RADIUS; r++) {
                // 取出目前這圈的左上, 上, 右上, 左, 右, 左下, 下, 右下的點的值
                // qwe    qwe
                // asd -> a d
                // zxc    zxc
                for (int u = i - r; u <= i + r; u += r) {
                    for (int v = j - r; v <= j + r; v += r) {
                        // 忽略中點 (s)
                        if (u == i && v == j) continue;
                        // 忽略超越邊界的點
                        if (u < 0 || v < 0 || u >= gray_mat.rows || v >= gray_mat.cols) continue;
                        // 加上合法點值和原始中點值的差
                        
                        //LAB方法
                        //L = L + (int)src_lab.at<Vec3b>(u, v)[0];
                        //A = A + (int)src_lab.at<Vec3b>(u, v)[1];
                        //B = B + (int)src_lab.at<Vec3b>(u, v)[2];
                        
                        processed_center_value += abs(center_value - gray_mat.at<uchar>(u, v));
                    }
                }
            }
            //LAB方法
            L = L / 104;
            A = A / 104;
            B = B / 104;
            int LL = (int)src_lab.at<Vec3b>(i, j)[0];
            int AA = (int)src_lab.at<Vec3b>(i, j)[1];
            int BB = (int)src_lab.at<Vec3b>(i, j)[2];
            //xxx.at<uchar>(i, j) = sqrt(pow(abs(LL - L),2) + pow(abs(AA - A),2) + pow(abs(BB - B),2));
            
            
            // cout << "i = "
            // cout << " processed = " << processed_center_value << endl;
            xxx.at<uchar>(i, j) = processed_center_value;
        }
    }
    

    
    // Canny 的方法
    //blur(src_gray, src_gray, Size(3,3));//模糊化
    //Canny(src_gray, canny_output, 50, 50*2, 3); ///檢測邊緣
    //cv::imshow("show canny", canny_output);
    //integral(canny_output, integral_output); //影像積分
    
    
    // Sobel 的方法
    /*medianBlur(src_gray, src_gray, 5);//模糊化
    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    
    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    
    cv::imshow("show sobel", grad);
    
    integral(grad, integral_output);*/
    
    cv::imshow("show sobel", xxx);
    integral(xxx, integral_output);
    
    
    //去除 integral運算 產生的一排0
    cv::Mat Image(integral_output);
    cv::Rect ImageROI(1, 1, integral_output.cols-1, integral_output.rows-1);
    integral_output1 = Image(ImageROI);
    
    MaskSizeX = getMaskSize(1, integral_output1.cols, integral_output1);
    //MaskSizeX =464; //349; 384
    //MaskSizeY =352; //341; 376
    //MaskSizeY = (MaskSizeX / 16) * 9;
    MaskSizeY = getMaskSize(0, integral_output1.rows, integral_output1);
    //if ((MaskSizeY * 2) <= MaskSizeX) {
    //    MaskSizeY = (MaskSizeX / 4) * 3;
    //}

    cout << "X = " << integral_output1.cols << endl;
    cout << "Y = " << integral_output1.rows << endl;
    cout << "MaskSizeX = " << MaskSizeX << endl;
    cout << "MaskSizeY = " << MaskSizeY << endl;

    canny_output_pix_size = MaskSizeY * MaskSizeX;
    for (int cropPositionY = 0; cropPositionY < (integral_output1.rows - MaskSizeY); cropPositionY += 1) {
        for (int cropPositionX = 0; cropPositionX < (integral_output1.cols - MaskSizeX); cropPositionX += 1) {
            
            /*float pix3 = (float)(*(integral_output1.data + integral_output1.step[0] * (cropPositionY + 300) + integral_output1.step[1] * (cropPositionX + 200)));
            float pix2 = (float)(*(integral_output1.data + integral_output1.step[0] * (cropPositionY + 300) + integral_output1.step[1] * cropPositionX));
            float pix1 = (float)(*(integral_output1.data + integral_output1.step[0] * cropPositionY + integral_output1.step[1] * (cropPositionX + 200)));
            float pix0 = (float)(*(integral_output1.data + integral_output1.step[0] * cropPositionY + integral_output1.step[1] * cropPositionX));*/
            int pix3 = static_cast<int>(integral_output1.at<int>(cropPositionY + MaskSizeY, cropPositionX + MaskSizeX));
            int pix2 = static_cast<int>(integral_output1.at<int>(cropPositionY, cropPositionX + MaskSizeX));
            int pix1 = static_cast<int>(integral_output1.at<int>(cropPositionY + MaskSizeY, cropPositionX));
            int pix0 = static_cast<int>(integral_output1.at<int>(cropPositionY, cropPositionX));

            int sumPix = pix3 + pix0 - pix1 - pix2;
            
            // 依照 pix 在 Mask 上的平均值為依據
            int temp2 = sumPix;
            
            //int temp2 = sumPix / canny_output_pix_size;
            
            //cout << "temp22 = " << temp2 << endl;
            //cout << "cropPositionY = " << cropPositionY << endl;
            //cout << "cropPositionX = " << cropPositionX << endl;
            if (temp < temp2) {
                // cout << "temp2 = " << temp2 << endl;
                temp = temp2;
                finalX = cropPositionX;
                finalY = cropPositionY;
            }
        }
    }
    
    final_origin_image = imread(origin_file, -1); // resoure image
    cv::Mat finalImage(final_origin_image);
    cout << "finalX = " << finalX << endl;
    cout << "finalY = " << finalY << endl;
    cv::Rect finalROI(finalX, finalY, MaskSizeX, MaskSizeY);
    
    cv::Mat showImage = finalImage(finalROI);
    cv::imshow("show crop image", showImage);
    cv::waitKey(0);
    return 0;
}

int getMaskSize(int isXorY, int listLength, Mat frame)
{
    long int sum = 0;
    float uSum = 0;
    //int x = -1;
    //int y = -1;
    //int finalx = 0;
    //int finaly = 0;
    int successive_count = 0;
    int temp_count = 0;
    
    const int m = listLength;
    int pixList[m];
    if (isXorY == 1) {
        for (int i=0; i < listLength; i++) {
            if (i == 0) {
                pixList[i] = static_cast<int>(frame.at<int>(frame.rows-1, i));
            } else {
                pixList[i] = static_cast<int>(frame.at<int>(frame.rows-1, i)) - static_cast<int>(frame.at<int>(frame.rows-1, i-1));
            }
        }
    } else {
        for (int i=0; i < listLength; i++) {
            if (i == 0) {
                pixList[i] = static_cast<int>(frame.at<int>(i, frame.cols-1));
            } else {
                pixList[i] = static_cast<int>(frame.at<int>(i, frame.cols-1)) - static_cast<int>(frame.at<int>(i-1, frame.cols-1));
            }
        }
    }
    
    
    const int ARRAY_SIZE = sizeof(pixList)/sizeof(pixList[0]);
    
    int he_out[ARRAY_SIZE];//陣列均衡化用的 Histogram equalization
    long long sd_out[ARRAY_SIZE];//標準差在用的 Standard Deviation
    float c_out[ARRAY_SIZE]; //畫線在用的
    
    //平滑化
    const short FILTER_SIZE = 9; /* 最好是奇數 */
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        float cell = 0;
        int actual_filter_size = 0;
        for (int j = i - FILTER_SIZE / 2; j <= i + FILTER_SIZE / 2; ++j) {
            if (j < 0 || j >= ARRAY_SIZE)
                continue;
            cell += pixList[j];
            actual_filter_size += 1;
        }
        cell /= actual_filter_size;
        he_out[i] = round(cell);
    }
    
    for (int i=0; i<=listLength; i++) {
        sum = sum + he_out[i];
    }
    
    for (int i=0; i<=listLength; i++) {
        float pp = he_out[i];
        float ww = sum;
        //uSum = uSum + ((he_out[i] / sum) * i);
        uSum = uSum + ((pp / ww) * i);
    }
    cout << "uSum = " << uSum << endl;
    
    //drew line
    for (int i=0; i<=listLength; i++) {
        c_out[i] = pixList[i];
    }
    
    //算標準差
    long long variance_sum_temp = 0;
    long long variance_sum_temp2 = 0 ;
    long long expected = sum/listLength;
    //long long uExpected = uSum/listLength;
    for (int i=0; i<listLength; i++) {
        //cout << "qoo = " << variance_sum_temp << endl;
        variance_sum_temp2 = variance_sum_temp2 + pow((i - int(uSum)), 2) * pixList[i] / sum;
    }
    
    for (int i=0; i<listLength; i++) {
        //cout << "qoo = " << variance_sum_temp << endl;
        variance_sum_temp = variance_sum_temp + pow((pixList[i] - expected), 2);
    }
    variance_sum_temp = variance_sum_temp / listLength;
    //cout << "variance / listlength = " << variance_sum_temp << endl;
    variance_sum_temp = sqrt(variance_sum_temp);
    cout << "^^variance = " << variance_sum_temp << endl;
    
    cout << "sum = " << sum << endl;
    cout << "sum/listLength = " << (sum/listLength) << endl;
    
    /*if (isXorY == 1) {
        int max_mum = 0;
        origin_image_line = imread(origin_file, -1); // resoure image
        for (int i=0; i<=listLength; i++) {
            if (c_out[i] > max_mum) {
                max_mum = c_out[i];
            }
        }
        cout << "max_mum = " << max_mum << endl;
        cout << "max_mum = " << round((c_out[0] / max_mum) * (origin_image_line.rows-1)) << endl;
        Point a;
        Point b;
        for (int i=0; i<=listLength; i++) {
            //cout << "pix = "<< pixList[i] << endl;
            a = Point (i, origin_image_line.rows - 1);
            b = Point (i, (origin_image_line.rows - 1) - round((c_out[i] / max_mum) * (origin_image_line.rows-1)));
            line(origin_image_line,a,b,Scalar(0,0,255));
        }
        cout << "qooqoooqooo = " << int(uSum) - int(sqrt(variance_sum_temp2)) << endl;
        Point r = Point (int(uSum) - int(sqrt(variance_sum_temp2)), 0);
        Point t = Point (int(uSum) - int(sqrt(variance_sum_temp2)), (origin_image_line.rows - 1));
        line(origin_image_line,r,t,Scalar(255,0,0));
        r = Point (int(uSum) + int(sqrt(variance_sum_temp2)), 0);
        t = Point (int(uSum) + int(sqrt(variance_sum_temp2)), (origin_image_line.rows - 1));
        line(origin_image_line,r,t,Scalar(0,255,0));
        
        cv::imshow("show line image", origin_image_line);
    }*/
    if (isXorY == 0) {
     int max_mum = 0;
     origin_image_line = imread(origin_file, -1); // resoure image
     for (int i=0; i<=listLength; i++) {
     if (c_out[i] > max_mum) {
     max_mum = c_out[i];
     }
     }
     cout << "max_mum = " << max_mum << endl;
     cout << "max_mum = " << round((c_out[0] / max_mum) * (origin_image_line.cols-1)) << endl;
     Point a;
     Point b;
     for (int i=0; i<=listLength; i++) {
     //cout << "pix = "<< pixList[i] << endl;
     a = Point (origin_image_line.cols - 1, i);
     b = Point ((origin_image_line.cols - 1) - round((c_out[i] / max_mum) * (origin_image_line.cols-1)), i);
     line(origin_image_line,a,b,Scalar(0,0,255));
     }
     cout << "qooqoooqooo = " << int(uSum) - int(sqrt(variance_sum_temp2)) << endl;
     Point r = Point (0, int(uSum) - int(sqrt(variance_sum_temp2)));
     Point t = Point ((origin_image_line.cols - 1) ,int(uSum) - int(sqrt(variance_sum_temp2)));
     line(origin_image_line,r,t,Scalar(255,0,0));
     r = Point (0, int(uSum) + int(sqrt(variance_sum_temp2)));
     t = Point ((origin_image_line.cols - 1), int(uSum) + int(sqrt(variance_sum_temp2)));
     line(origin_image_line,r,t,Scalar(0,255,0));
     
     cv::imshow("show line image", origin_image_line);
     }
    
    
    /////////舊方法
    /*for (int i=0; i<listLength; i++) {
        //cout << b_out[i] << endl;
        if (he_out[i] >= (expected - variance_sum_temp)) {
            successive_count = successive_count + 1;
        }
        if (he_out[i] < (expected - variance_sum_temp)) {
            if (temp_count < successive_count) {
                temp_count= successive_count;
                successive_count = 0;
            }
        }
        if (temp_count == 0 && i+1 == listLength) {
            temp_count = successive_count;
        }
    }*/
    
    cout << "uper = " << (int(uSum) + int(sqrt(variance_sum_temp2))) << endl;
    cout << "dow = " << (int(uSum) - int(sqrt(variance_sum_temp2))) << endl;
    
    temp_count = (int(uSum) + int(sqrt(variance_sum_temp2))) - (int(uSum) - int(sqrt(variance_sum_temp2)));
    cout << "temp_count = " << temp_count << endl;
    //temp_count = temp_count + (temp_count * 0.6);
    return temp_count;
}

