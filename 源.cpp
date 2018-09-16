//#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
//#include <vector>       
//
using namespace cv;
using namespace std;
//
//int main(int argc, char** argv)
//{
//	cv::Mat inp_img = cv::imread("house.jpg");
//	if (!inp_img.data) {
//		cout << "Something Wrong";
//		return -1;
//	}
//	namedWindow("Input Image", CV_WINDOW_AUTOSIZE);
//	cv::imshow("Input Image", inp_img);
//
//	cv::Mat clahe_img;
//	cv::cvtColor(inp_img, clahe_img, CV_BGR2Lab);
//	std::vector<cv::Mat> channels(3);
//	cv::split(clahe_img, channels);
//
//	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
//	clahe->setClipLimit(4);
//	cv::Mat dst;
//	clahe->apply(channels[0], dst);
//	dst.copyTo(channels[0]);
//	cv::merge(channels, clahe_img);
//
//	cv::Mat image_clahe;
//	cv::cvtColor(clahe_img, image_clahe, CV_Lab2BGR);
//
//	namedWindow("CLAHE Image", CV_WINDOW_AUTOSIZE);
//	cv::imshow("CLAHE Image", image_clahe);
//	imwrite("out.jpg", image_clahe);
//	cv::waitKey(0);
//	destroyAllWindows();
//
//	return 0;
//}
Mat histImage(500, 500, CV_8UC3, cv::Scalar(0, 0, 0));
void huiduzhifangtu(Mat ImageGray)
{


	// ����ֱ��ͼ����
	const int channels[1] = { 0 };
	const int histSize[1] = { 256 };
	float pranges[2] = { 0,255 };
	const float* ranges[1] = { pranges };



	cv::MatND hist;
	// ����ֱ��ͼ
	cv::calcHist(&ImageGray, 1, channels, cv::Mat(), hist, 1,
		histSize, ranges);
	// ��ʼ����������
	int hist_w = 500;
	int hist_h = 500;
	int nHistSize = 255;
	//// ����
	int bin_w = cvRound((double)hist_w / nHistSize);
	// ��ֱ��ͼ��һ������Χ [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows,
		cv::NORM_MINMAX, -1, cv::Mat());
	// ��ֱ��ͼ�����ϻ���ֱ��ͼ
	for (int i = 1; i < nHistSize; i++)
	{
		line(histImage, cv::Point(bin_w*(i - 1),
			hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w*(i),
				hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}


}

int main(int argc, char** argv)
{
	cv::Mat inp_img = cv::imread("fog1.jpg");
	if (!inp_img.data) {
		cout << "Something Wrong";
		return -1;
	}
	namedWindow("Input Image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Input Image", inp_img);
	GaussianBlur(inp_img, inp_img, Size(1, 1),0,0);
	cv::Mat clahe_img;
	cv::cvtColor(inp_img, clahe_img, CV_BGR2Lab);
	std::vector<cv::Mat> channels(3);
	cv::split(clahe_img, channels);

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	// ֱ��ͼ�����Ӹ߶ȴ��ڼ�����ClipLimit�Ĳ��ֱ��ü�����Ȼ����ƽ�����������ֱ��ͼ  
	// �Ӷ���������ͼ��  
	clahe->setClipLimit(4.); //(int)(4.*(8*8)/256)  
	clahe->setTilesGridSize(Size(8, 8)); // ��ͼ���Ϊ8*8��  
	cv::Mat dst;
	clahe->apply(channels[0], dst);
	dst.copyTo(channels[0]);
	cv::merge(channels, clahe_img);

	cv::Mat image_clahe;
	cv::cvtColor(clahe_img, image_clahe, CV_Lab2BGR);

	//cout << cvFloor(-1.5) << endl;  

	namedWindow("CLAHE Image", CV_WINDOW_AUTOSIZE);
	cv::imshow("CLAHE Image", image_clahe);
	//GaussianBlur(image_clahe, image_clahe, Size(5, 5), 0, 0);
	huiduzhifangtu(image_clahe);
	imshow("claheֱ��ͼ", histImage);
	imwrite("out.jpg", histImage);
	cv::waitKey(0);
	destroyAllWindows();

	return 0;
}