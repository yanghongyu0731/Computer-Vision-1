/************************************************************************/
/*   ·¨¶£³ë R09921012							                        */
/*   2020/9/21                                                          */
/************************************************************************/

#include <fstream>       

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int main()
{

	//Part1.

	//(a)
	Mat img = imread("lena.bmp", CV_8UC1);
	imshow("original", img);
	int center_i = img.rows / 2;
	int center_j = img.cols / 2;
	for (int i = 0; i < center_i; i++) { //change rows data
		for (int j = 0; j < img.cols; j++){
			uchar temp = img.at<uchar>(i,j);
			img.at<uchar>(i,j) = img.at<uchar>(img.rows - i - 1,j);
			img.at<uchar>(img.rows - i - 1,j) = temp;
		}
	}
	imshow("upside-down lena", img);
	imwrite("upside-down lena.jpg", img);

	//(b)
	img = imread("lena.bmp", CV_8UC1);
	center_i = img.rows / 2;
	center_j = img.cols / 2;
	for (int i = 0; i < img.rows; i++) { //change cols data
		for (int j = 0; j < center_j; j++) {
			uchar temp = img.ptr<uchar>(i)[j];
			img.ptr<uchar>(i)[j] = img.ptr<uchar>(i)[img.cols - j - 1];
			img.ptr<uchar>(i)[img.cols - j - 1] = temp;
		}
	}
	imshow("rightside-left", img);
	imwrite("rightside-left.jpg", img);

	//(c)
	img = imread("lena.bmp", CV_8UC1);
	center_i = img.rows / 2;
	center_j = img.cols / 2;
	for (int i = 0; i < img.rows; i++) { //change rows&cols data
		for (int j = 0; j < i; j++) {
			uchar temp = img.ptr<uchar>(i)[j];
			img.ptr<uchar>(i)[j] = img.ptr<uchar>(j)[i];
			img.ptr<uchar>(j)[i] = temp;
		}
	}
	imshow("diagonally mirrowed", img);
	imwrite("diagonally mirrowed.jpg", img);

	//Part2.

	//(d)
	img = imread("lena.bmp", CV_8UC1);
	center_i = img.rows / 2;
	center_j = img.cols / 2;
	Point2f a(399, 399); //rotate center
	Mat rot_mat(2, 3, CV_32FC1);
	rot_mat = getRotationMatrix2D(a, -45, 1); //find rotation tf matrix
	Size dsize(800, 800); //new img size
	
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, 400-256, 
										   0, 1, 400-256); // set transfermation matrix
	warpAffine(img, img, trans_mat, dsize); //transfer
	warpAffine(img, img, rot_mat, dsize); //rotate


	for (int i = 0; i < 800; i++) {
		for (int j = 0; j < 800; j++) {
			if (img.ptr<uchar>(i)[j] == 0)img.ptr<uchar>(i)[j] = 255;
		}
	}

	imshow("rotate", img);
	imwrite("rotate.jpg", img);

	//(e)
	img = imread("lena.bmp", CV_8UC1);
	resize(img, img,Size(img.rows/2,img.cols/2), 0, 0); //change img size to half

	imshow("shrink", img);
	imwrite("shrink.jpg", img);

	//(f)
	img = imread("lena.bmp", CV_8UC1);

	std::fstream file;
	file.open("binarize.txt", std::ios::out | std::ios::trunc); //write binarize.txt file
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.ptr<uchar>(i)[j] > 128) { //set threadhold
				file.write("1", 1);
				img.ptr<uchar>(i)[j] = 255;
			}
			else { 
				file.write("0", 1);
				img.ptr<uchar>(i)[j] = 0;
			}
		}
		file.write("\n", 1);
	}
	file.close();
	imshow("binary", img);
	imwrite("binary.jpg", img);



	waitKey(0);
	return 0;
}