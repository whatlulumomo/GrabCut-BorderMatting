#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;

struct Info {
	Vec3b backMean;
	Vec3b frontMean;
	double backVar;
	double frontVar;
};

class  point {
public:
	int x;
	int y;
	int delta, sigma;
	int dis;
	double alpha;
	Info nearbyInfo;
	point(int x, int y, int dis = 0)
	{
		this->x = x;
		this->y = y;
		this->dis = dis;
	}
	point() {};
	int distance(point t)
	{
		return sqrt((this->x - t.x)*(this->x - t.x) + (this->y - t.y)*(this->y - t.y));
	}
};

class contourPoint {
public:
	vector<point> neighbor;
	point pointInfo;
	contourPoint(point p) { this->pointInfo = p; };
};