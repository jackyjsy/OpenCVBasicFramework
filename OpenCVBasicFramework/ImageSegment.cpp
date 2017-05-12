#include "ImageSegment.h"
#include<numeric>
#include<cmath>

using namespace cv;
using namespace std;

vector<int> sort_indexes(const vector<double> v)
{

	// initialize original index locations
	vector<int> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2]; });
	return idx;
}

ImageSegment::ImageSegment(cv::Mat src)
{
	int clusters = 3;
	ImageSegment(src, clusters);
}

ImageSegment::ImageSegment(cv::Mat src, int clusters)
{
	Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y, x)[z];

	ClusterCount = clusters;
	Mat labels;
	int attempts = 5;
	Mat centers;
	vector<int> numbers(ClusterCount, 0);
	kmeans(samples, ClusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	Mat new_image(src.size(), src.type());
	cv::Mat image_labels(src.rows, src.cols, CV_8U);
	
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*src.rows, 0);
			image_labels.at<uchar>(y, x) = cluster_idx;
			numbers[cluster_idx]++;
			new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}
	new_image.copyTo(Segments);
	centers.copyTo(Centers);
	image_labels.copyTo(Label);
	ClusterWeight = numbers;
}

Scalar ImageSegment::getMedianColor(int index_start, int index_end)
{
	vector<double> center_y(ClusterCount);
	for (int i = 0; i < ClusterCount; i++)
	{
		float* ptr = Centers.ptr<float>(i);
		center_y[i] = 0.229*(double)ptr[2] + 0.587*(double)ptr[1] + 0.114*(double)ptr[0];
	}
	vector<int> sorted_indexes = sort_indexes(center_y);
	//SortedIndexes = sorted_indexes;
	Scalar result(0, 0, 0);
	double total_weight = 0;
	for (int i = index_start - 1; i < index_end; i++)
	{

		int index_unsorted = sorted_indexes[i];
		total_weight += ClusterWeight[index_unsorted];
		float* ptr = Centers.ptr<float>(index_unsorted);
		result += Scalar((double)ptr[0], (double)ptr[1], (double)ptr[2])*ClusterWeight[index_unsorted];
	}

	result /= total_weight;
	return result;
}

