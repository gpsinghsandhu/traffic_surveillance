#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class vehicle_detector {
public:
	CascadeClassifier classifier;
	vector<Rect> detections;
	bool valid;
	int min_neighbours;
	double window_scale_factor;
	Size min_size, max_size, window_size;
	Mat gray;

	vehicle_detector();
	vehicle_detector(String path, Size _window_size, double _scale_factor,
			int _min_neighbours, Size _minSize, Size _maxSize);
	void detect(Mat &img);
};

vehicle_detector::vehicle_detector(void) {
	window_scale_factor = 0.0;
	min_neighbours = 0;
	valid = false;
}

vehicle_detector::vehicle_detector(String _path, Size _window_size, double _scale_factor = 1.1,
		int _min_neighbours = 30, Size _min_size = Size(), Size _max_size = Size()) {
	classifier.load(_path);
	window_size = _window_size;
	window_scale_factor = _scale_factor;
	min_neighbours = _min_neighbours;
	min_size = _min_size;
	max_size = _max_size;

	valid = true;
}

void vehicle_detector::detect(Mat &img) {
	if(img.type() != CV_8UC1)
		cvtColor(img, gray, COLOR_BGR2GRAY);
	else
		img.copyTo(gray);

	resize(gray, gray, window_size);
	classifier.detectMultiScale(gray, detections, window_scale_factor,
			min_neighbours, 0, min_size, max_size);
}
