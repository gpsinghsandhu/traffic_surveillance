#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

struct ctt{
	int vmin, vmax, smin;
	int hsize;
	const float* phranges;

	ctt() {
		vmin = 10, vmax = 256, smin = 30;
		hsize = 32;
		float hranges[] = {0,180};
		phranges = {hranges};
	}
} constant;

class vehicle_model {
public :
	Rect rec;
	bool empty;
	Mat hsv, mask, hue, hist, hist_s, sat, val, hist_v;
	vector<Point> trajectory;

	vehicle_model(void);
	vehicle_model(Mat &img, Rect rect);
	bool is_empty(void);
	void push_trajectory_point(Point p);
};

vehicle_model::vehicle_model(void) {
	empty = true;
}

vehicle_model::vehicle_model(Mat &img, Rect rect) {
	rec = rect;
	trajectory.clear();
	trajectory.push_back(Point(rect.x, rect.y));
	cvtColor(img, hsv, COLOR_BGR2HSV);
	//img.copyTo(hsv);
	int _vmin = constant.vmin, _vmax = constant.vmax;

	inRange(hsv, Scalar(0, constant.smin, MIN(_vmin,_vmax)),
			Scalar(180, 256, MAX(_vmin, _vmax)), mask);
	int ch[] = {0, 0};
	hue.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &hue, 1, ch, 1);
	int ch1[] = {1, 0};
	sat.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &sat, 1, ch1, 1);
	int ch2[] = {2, 0};
	val.create(hsv.size(), hsv.depth());
	mixChannels(&hsv, 1, &val, 1, ch2, 1);

	Mat roi(hue, rec), maskroi(mask, rec);
	Mat roi_s(sat, rec);
	Mat roi_v(val, rec);
	float hranges[] = {0,180};
	float sranges[] = {0,255};
	const float *phranges = {hranges};
	const float *psranges = {sranges};
	calcHist(&roi, 1, 0, maskroi, hist, 1, &constant.hsize, &phranges);
	calcHist(&roi_s, 1, 0, maskroi, hist_s, 1, &constant.hsize, &psranges);
	calcHist(&roi_v, 1, 0, maskroi, hist_v, 1, &constant.hsize, &psranges);

	normalize(hist, hist, 0, 255, NORM_MINMAX);
	normalize(hist_s, hist_s, 0, 255, NORM_MINMAX);
	normalize(hist_v, hist_v, 0, 255, NORM_MINMAX);
	empty = false;
}

bool vehicle_model::is_empty(void) {
	return empty;
}

void vehicle_model::push_trajectory_point(Point p) {
	trajectory.push_back(p);
}

