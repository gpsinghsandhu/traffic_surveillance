#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BG_VAL 255

Size frame_size = Size(320, 240);

//Some constants for the algorithm
const double pi = 3.142;
const double cthr = 0.00001;
const double alpha = 0.002;
const double cT = 0.05;
const double covariance0 = 11.0;
const double cf = 0.1;
const double cfbar = 1.0-cf;
const double temp_thr = 9.0*covariance0*covariance0;
const double prune = -alpha*cT;
const double alpha_bar = 1.0-alpha;
//Temperory variable
int overall = 0;

//Structure used for saving various components for each pixel
struct gaussian
{
	double mean[3], covariance;
	double weight;								// Represents the measure to which a particular component defines the pixel value
	gaussian* Next;
	gaussian* Previous;
} *ptr, *start, *rear, *g_temp, *save, *next, *previous, *nptr, *temp_ptr;

struct Node
{
	gaussian* pixel_s;
	gaussian* pixel_r;
	int no_of_components;
	Node* Next;
} *N_ptr, *N_start, *N_rear;


struct Node1
{
	cv::Mat gauss;
	int no_of_comp;
	Node1* Next;
} *N1_ptr, *N1_start, *N1_rear;

//Some function associated with the structure management
Node* Create_Node(double info1, double info2, double info3);
void Insert_End_Node(Node* np);
gaussian* Create_gaussian(double info1, double info2, double info3);

Node* Create_Node(double info1, double info2, double info3)
{
	N_ptr = new Node;
	if(N_ptr != NULL)
	{
		N_ptr->Next = NULL;
		N_ptr->no_of_components = 1;
		N_ptr->pixel_s = N_ptr->pixel_r = Create_gaussian(info1,info2, info3);
	}
	return N_ptr;
}

gaussian* Create_gaussian(double info1, double info2, double info3)
{
	ptr = new gaussian;
	if(ptr != NULL)
	{
		ptr->mean[0] = info1;
		ptr->mean[1] = info2;
		ptr->mean[2] = info3;
		ptr->covariance = covariance0;
		ptr->weight = alpha;
		ptr->Next = NULL;
		ptr->Previous = NULL;
	}
	return ptr;
}

void Insert_End_Node(Node* np)
{
	if( N_start != NULL )
	{
		N_rear->Next = np;
		N_rear = np;
	}
	else
		N_start = N_rear = np;
}

void Insert_End_gaussian(gaussian* nptr)
{
	if(start != NULL)
	{
		rear->Next = nptr;
		nptr->Previous = rear;
		rear = nptr;
	}
	else
		start = rear = nptr;
}

gaussian* Delete_gaussian(gaussian* nptr)
{
	previous = nptr->Previous;
	next = nptr->Next;
	if(start != NULL)
	{
		if(nptr == start && nptr == rear)
		{
			start = rear = NULL;
			delete nptr;
		}
		else if(nptr == start)
		{
			next->Previous = NULL;
			start = next;
			delete nptr;
			nptr = start;
		}
		else if(nptr == rear)
		{
			previous->Next = NULL;
			rear = previous;
			delete nptr;
			nptr = rear;
		}
		else
		{
			previous->Next = next;
			next->Previous = previous;
			delete nptr;
			nptr = next;
		}
	}
	else
	{
		std::cout << "Underflow........";
		//getch();
		exit(0);
	}
	return nptr;
}

class BG_sub {
public :
	bool init;
	int img_rows, img_cols;
	int img_type;
	int i, j, k, nL, nC, background;
	Mat bin_img;
	double sum, sum1;
	double temp_cov;
	double weight;
	double var;
	double mult, mal_dist;
	double muR,muG,muB,dR,dG,dB,rVal,gVal,bVal;
	bool close;

	int dilation_size;
	Mat morph_element, temp;

	Vec3f val;
	uchar* r_ptr;
	uchar* b_ptr;

	bool initialize(Mat &img) {
		img_rows = img.rows, img_cols  = img.cols;
		img_type = img.type();
		if(img_type != CV_8UC3) {
			cout << "Only CV_8UC3 type supported till now\n";
			return false;
		}
		for( i=0; i<img_rows; i++ ) {
			r_ptr = img.ptr(i);
			for( j=0; j<img_cols; j++ ) {
				N_ptr = Create_Node(*r_ptr,*(r_ptr+1),*(r_ptr+2));
				if( N_ptr != NULL ) {
					N_ptr->pixel_s->weight = 1.0;
					Insert_End_Node(N_ptr);
				}
				else {
					std::cout << "Memory limit reached... can't create all gaussians";
					return false;
				}
			}
		}
		if(img.isContinuous() == true) {
			nL = 1;
			nC = img_rows*img_cols*img.channels();
		}
		else {
			nL = img_rows;
			nC = img_cols*img.channels();
		}
		bin_img = Mat(img_rows, img_cols, CV_8UC1, Scalar(0,0,0));
		sum = 0.0, sum1 = 0.0;
		temp_cov = 0.0;
		weight = 0.0;
		var = 0.0;
		dilation_size = 5;
		morph_element = getStructuringElement( MORPH_ELLIPSE,
												Size( 2*dilation_size + 1, 2*dilation_size+1 ),
												Point( dilation_size, dilation_size ) );
		init = true;
		return true;
	}

	void preprocess(Mat &input, Mat &output) {
		resize(input, output, frame_size);
		GaussianBlur(output, output, Size(3,3), 1.0, 1.0);
	}

	void postprocess(Mat &input, Mat &output) {
		morphologyEx(input, output, MORPH_CLOSE, morph_element, Point(-1,-1));
	}

	bool subtract_BG(Mat &img) {
		if(!init) {
			cout << "Model not initialized yet\n";
			return false;
		}
		if(img.rows != img_rows || img.cols != img_cols) {
			cout << "Dimensions of image and model do not match\n";
			return false;
		}
		preprocess(img, temp);
		//cv::resize(orig_img,orig_img,cv::Size(340,260));
		//cv::Mat result(bin_img.size(),CV_8U,cv::Scalar(255));

		//cv::cvtColor(orig_img, orig_img, CV_BGR2YCrCb);
		//cv::GaussianBlur(orig_img, orig_img, cv::Size(3,3), 3.0);

		//cv::cvtColor(bin_img, bin_img, CV_RGB2GRAY);

		N_ptr = N_start;
		for( i=0; i<nL; i++) {
			r_ptr = temp.ptr(i);
			b_ptr = bin_img.ptr(i);
			for( j=0; j<nC; j+=3) {
				sum = 0.0;
				sum1 = 0.0;
				close = false;
				background = BG_VAL;


				rVal = *(r_ptr++);
				gVal = *(r_ptr++);
				bVal = *(r_ptr++);

				start = N_ptr->pixel_s;
				rear = N_ptr->pixel_r;
				ptr = start;

				temp_ptr = NULL;

				if(N_ptr->no_of_components > 4) {
					Delete_gaussian(rear);
					N_ptr->no_of_components--;
				}

				for( k=0; k<N_ptr->no_of_components; k++ ) {


					weight = ptr->weight;
					mult = alpha/weight;
					weight = weight*alpha_bar + prune;
					if(close == false) {
						muR = ptr->mean[0];
						muG = ptr->mean[1];
						muB = ptr->mean[2];

						dR = rVal - muR;
						dG = gVal - muG;
						dB = bVal - muB;

						/*del[0] = value[0]-ptr->mean[0];
						del[1] = value[1]-ptr->mean[1];
						del[2] = value[2]-ptr->mean[2];*/


						var = ptr->covariance;

						mal_dist = (dR*dR + dG*dG + dB*dB);

						if((sum < cfbar) && (mal_dist < 16.0*var*var))
								background = 255-BG_VAL;

						if( mal_dist < 9.0*var*var) {
							weight += alpha;
							//mult = mult < 20.0*alpha ? mult : 20.0*alpha;

							close = true;
							ptr->mean[0] = muR + mult*dR;
							ptr->mean[1] = muG + mult*dG;
							ptr->mean[2] = muB + mult*dB;
							//if( mult < 20.0*alpha)
							//temp_cov = ptr->covariance*(1+mult*(mal_dist - 1));
							temp_cov = var;// + mult*(mal_dist - var);
							ptr->covariance = temp_cov<5.0?5.0:(temp_cov>20.0?20.0:temp_cov);
							temp_ptr = ptr;
						}
					}

					if(weight < -prune) {
						ptr = Delete_gaussian(ptr);
						weight = 0;
						N_ptr->no_of_components--;
					}
					else {
					//if(ptr->weight > 0)
						sum += weight;
						ptr->weight = weight;
					}
					ptr = ptr->Next;
				}

				if( close == false ) {
					ptr = new gaussian;
					ptr->weight = alpha;
					ptr->mean[0] = rVal;
					ptr->mean[1] = gVal;
					ptr->mean[2] = bVal;
					ptr->covariance = covariance0;
					ptr->Next = NULL;
					ptr->Previous = NULL;
					//Insert_End_gaussian(ptr);
					if(start == NULL)
						// ??
						start = rear = NULL;
					else {
						ptr->Previous = rear;
						rear->Next = ptr;
						rear = ptr;
					}
					temp_ptr = ptr;
					N_ptr->no_of_components++;
				}

				ptr = start;
				while( ptr != NULL) {
					ptr->weight /= sum;
					ptr = ptr->Next;
				}

				while(temp_ptr != NULL && temp_ptr->Previous != NULL) {
					if(temp_ptr->weight <= temp_ptr->Previous->weight)
						break;
					else {
						//count++;
						next = temp_ptr->Next;
						previous = temp_ptr->Previous;
						if(start == previous)
							start = temp_ptr;
						previous->Next = next;
						temp_ptr->Previous = previous->Previous;
						temp_ptr->Next = previous;
						if(previous->Previous != NULL)
							previous->Previous->Next = temp_ptr;
						if(next != NULL)
							next->Previous = previous;
						else
							rear = previous;
						previous->Previous = temp_ptr;
					}
					temp_ptr = temp_ptr->Previous;
				}
				N_ptr->pixel_s = start;
				N_ptr->pixel_r = rear;

				*b_ptr++ = background;
				N_ptr = N_ptr->Next;
			}

		}
		postprocess(bin_img, bin_img);
		return true;
	}

	void get_BG_img(Mat &img) {
		if(!init) {
			cout << "Model not initialized yet - can't produce BG img\n";
			return;
		}
		if(!img.data) {
			img = Mat(img_rows, img_cols, CV_8UC3, Scalar(0,0,0));
		}
		//check
		N_ptr = N_start;
		for( i=0; i<nL; i++) {
			r_ptr = img.ptr(i);
			for( j=0; j<nC; j+=3) {
				start = N_ptr->pixel_s;
				ptr = start;
				int temp = int(ptr->mean[0]);
				*r_ptr = uchar(temp);
				//cout << ptr->mean[0] << " : "  << temp << " : " << int(*r_ptr) << "\n";
				r_ptr++;
				temp = int(ptr->mean[1]);
				*r_ptr= uchar(temp);
				r_ptr++;
				*r_ptr = uchar(temp);
				r_ptr++;
				N_ptr = N_ptr->Next;
			}
		}
	}
};
