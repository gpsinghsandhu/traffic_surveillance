#include <opencv2/opencv.hpp>
#include <iostream>

#include "vehicle_model.hpp"
#include "vehicle_detection.hpp"
#include "BG_subtraction.hpp"

using namespace std;
using namespace cv;

//--Constants--//
Size im_size = Size(320, 240);

//--Classifier constants--//
int min_neighbours = 20;
double scale_factor = 1.1;
Size min_size = Size(), max_size = Size();
Size window_size = im_size;

struct vehicle_list {
	vehicle_model *model;
	vehicle_list *next, *prev;

	vehicle_list(Mat &, Rect);
} *vehicles;

vehicle_list::vehicle_list(Mat &img, Rect rec) {
	model = new vehicle_model(img, rec);
	next = prev = NULL;
}

vehicle_list* push_vehicle(vehicle_list* head, vehicle_list* x) {
	if(head == NULL)
		return x;
	x->next = head;
	head->prev = x;
	head = x;
	return head;
}

vehicle_list* delete_vehicle(vehicle_list* head, vehicle_list *x) {
	if(head == NULL)
		return head;
	vehicle_list *p, *n;
	p = x->prev;
	n = x->next;

	if(p != NULL)
		p->next = n;
	if(n != NULL)
		n->prev = p;

	if(head == x)
		head = n;
	delete(x);
	return head;
}

VideoCapture cap;
vehicle_detector *detector;
BG_sub BG_model;

//--Functions--//
void init(void);
void preprocess(Mat &img, Mat &processed);
vehicle_list* track_vehicles(vehicle_list *vehicles, vector<int> outgoing_vehicles, Mat &img);
vehicle_list* check_new_detections(vehicle_list *vehicles, vector<Rect> &detections, Mat &img);
vehicle_list* delete_vehicles(vehicle_list* vehicles, vector<int> outgoing_vehicles);
void draw_vehicles(vehicle_list *vehicles, Mat &img, Mat &output);
void draw_detections(Mat &, vector<Rect>);

int main() {
	Mat img, processed, output;
	int detection_rate = 0;;
	vector<int> outgoing_vehicles;

	init();
	cap.read(img);
	if(!BG_model.initialize(img)) {
		cout << "BG_model not initialized successfully\n";
		exit(0);
	}
	int count = 0;
	double duration = static_cast<double>(cv::getTickCount());
	while(cap.read(img)) {
		count++;
		preprocess(img, processed);
		vehicles = track_vehicles(vehicles, outgoing_vehicles, img);
		if(detection_rate == 0) {
			detection_rate = 2;
			detector->detect(img);
			vehicles = check_new_detections(vehicles, detector->detections, processed);
			draw_detections(processed, detector->detections);
		}
		//vehicles = delete_vehicles(vehicles, outgoing_vehicles);
		draw_vehicles(vehicles, processed, output);
		char ch = waitKey(10);
		if(ch == 27)
			break;
		else if(ch == ' ')
			waitKey(0);
		detection_rate--;
	}
	duration -= static_cast<double>(cv::getTickCount());
	duration /= (-cv::getTickFrequency());
	cout << "Total Frames : " << count << "\n";
	cout << "Duration : " << duration << "secs \n";
	cout << "Frame Rate : " << (double)count/duration << "frames/sec \n";
	return 0;
}

String video_path = "/home/guru/versionControl/8_xvid.avi";
String vehicle_classifier_path = "/home/guru/versionControl/final_project/carDetect/try_1/classifier4/cascade.xml";
void init(void) {
	// Initialize VideoCapture object
	cap.open(video_path);
	// Initialize detector
	detector = new vehicle_detector(vehicle_classifier_path, window_size, scale_factor,
			                         min_neighbours, min_size, max_size);
}

void preprocess(Mat &img, Mat& processed) {
	resize(img, processed, im_size);
}


Rect trackWindow;
vehicle_list* track_vehicles(vehicle_list *vehicles, vector<int> outgoing_vehicles, Mat &img) {
	vehicle_list *current_vehicle = vehicles;
	Mat hsv, mask, hue, hist, backproj, backproj_s, sat, hist_s, val, hist_v, backproj_v, bg_mask;
	BG_model.subtract_BG(img);
	BG_model.bin_img.copyTo(bg_mask);
	imshow("background subtraction", bg_mask);
	Rect rec;
	while(current_vehicle != NULL) {
		rec = vehicles->model->rec;
		cvtColor(img, hsv, COLOR_BGR2HSV);
		//img.copyTo(hsv);
		int _vmin = constant.vmin, _vmax = constant.vmax;

		inRange(hsv, Scalar(0, constant.smin, MIN(_vmin,_vmax)),
				Scalar(180, 256, MAX(_vmin, _vmax)), mask);
		//imshow("mask", mask);
		int ch[] = {0, 0};
		hue.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &hue, 1, ch, 1);
		int ch1[] = {1, 0};
		sat.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &sat, 1, ch1, 1);
		int ch2[] = {2, 0};
		val.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &val, 1, ch2, 1);
		hist = current_vehicle->model->hist;
		hist_s = current_vehicle->model->hist_s;
		hist_v = current_vehicle->model->hist_v;

		trackWindow = rec;
		float hranges[] = {0,180};
		float sranges[] = {0,255};
		const float *phranges = {hranges};
		const float *psranges = {sranges};
		calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
		calcBackProject(&sat, 1, 0, hist_s, backproj_s, &psranges);
		calcBackProject(&val, 1, 0, hist_v, backproj_v, &psranges);
		multiply(backproj, backproj_s, backproj, 1./255.0);
		//multiply(backproj, backproj_v, backproj, 1./255.0);
		//backproj &= mask;
		backproj &= bg_mask;
		//imshow("backproj", backproj);
		//waitKey(5);
		//RotatedRect trackBox = CamShift(backproj, trackWindow,
		//					TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 20, 1 ));
		meanShift(backproj, trackWindow,
									TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
		/*if( trackWindow.area() <= 1 )
		{
			int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
			trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
							   trackWindow.x + r, trackWindow.y + r) &
						  Rect(0, 0, cols, rows);
		}*/
		current_vehicle->model->rec = trackWindow;
		vehicle_list *x = current_vehicle;
		current_vehicle = current_vehicle->next;
		if(trackWindow.x < 10 || trackWindow.y+trackWindow.height > im_size.height-10)
			vehicles = delete_vehicle(vehicles, x);
	}
	return vehicles;
}

vehicle_list* check_new_detections(vehicle_list *vehicles, vector<Rect> &detections, Mat &img) {
	vehicle_list *current_vehicle = vehicles;
	vehicle_list *temp;

	while(current_vehicle != NULL) {
		temp = current_vehicle;
		current_vehicle = current_vehicle->next;
		delete(temp);
	}
	vehicles = NULL;
	for(int i=0; i<(int)detections.size(); i++) {
		Rect rec = detections[i];
		current_vehicle = new vehicle_list(img, rec);
		vehicles = push_vehicle(vehicles, current_vehicle);
	}
	return vehicles;
}

vehicle_list* delete_vehicles(vehicle_list *vehicles, vector<int> outgoing_vehicles) {
	return vehicles;
}

void draw_vehicles(vehicle_list *vehicles, Mat &img, Mat &output) {
	img.copyTo(output);
	vehicle_list *current = vehicles;

	while(current != NULL) {
		rectangle(output, current->model->rec, Scalar(255, 0, 0), 2);
		current = current->next;
	}

	imshow("original", img);
	imshow("output", output);
}

void draw_detections(Mat &img, vector<Rect> detections) {
	Mat out;
	img.copyTo(out);
	for(int i=0; i<(int)detections.size(); i++) {
		rectangle(out, detections[i], Scalar(0,255,0), 2);
	}
	imshow("detections", out);
	//waitKey(10);
}
