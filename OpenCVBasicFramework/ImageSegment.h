#pragma once
#include<opencv2/opencv.hpp>

class ImageSegment
{
public:
	//ImageSegment();
	ImageSegment(cv::Mat src);
	ImageSegment(cv::Mat src, int clusters);
	cv::Mat Label;
	cv::Mat Centers;
	cv::Mat Segments;
	int ClusterCount;
	std::vector<int> ClusterWeight;
	//std::vector<int> SortedIndexes;
	cv::Scalar getMedianColor(int index_start, int index_end);
	//cv::TermCriteria SegmentCriteria;
	//void set(cv::Mat);
};
