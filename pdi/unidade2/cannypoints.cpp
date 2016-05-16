#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <cstdlib>

using namespace std;
using namespace cv;

#define STEP 5
#define JITTER 3
#define RAIO 4

int top_slider = 100;
int top_slider_max = 200;

int width, height;
int x, y;

vector<int> yrange;
vector<int> xrange;

Vec3b val;

Mat image, imageGray, border, points, pointsCanny, pointsFinal;

char TrackbarName[50];

void on_trackbar_canny(int, void*);


int main(int argc, char** argv){
 
  image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
  imageGray = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);

  srand(time(0));
  
  if(!image.data){
	cout << "nao abriu" << argv[1] << endl;
    cout << argv[0] << " imagem";
    exit(0);
  }

  width=image.size().width;
  height=image.size().height;

  xrange.resize(height/STEP);
  yrange.resize(width/STEP);
  
  iota(xrange.begin(), xrange.end(), 0); 
  iota(yrange.begin(), yrange.end(), 0);

  for(uint i=0; i<xrange.size(); i++){
    xrange[i]= xrange[i]*STEP+STEP/2;
  }
  for(uint i=0; i<yrange.size(); i++){
    yrange[i]= yrange[i]*STEP+STEP/2;
  }  

  points = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

  random_shuffle(xrange.begin(), xrange.end());
  
  for(auto i : xrange){
    //random_shuffle(yrange.begin(), yrange.end());
    for(auto j : yrange){
      x = i+rand()%(2*JITTER)-JITTER+1;
      y = j+rand()%(2*JITTER)-JITTER+1;

      val[0] = (int)(image.at<Vec3b>(x,y)[0]);
      val[1] = (int)(image.at<Vec3b>(x,y)[1]);
      val[2] = (int)(image.at<Vec3b>(x,y)[2]);

      circle(points,
             Point(y,x),
             RAIO,
             Scalar(val[0], val[1], val[2]),
             -1,
             CV_AA);
    }
  }
  imshow("Pontilhismo sem o uso de Canny.png", points);

  sprintf(TrackbarName, "Threshold inferior");

  namedWindow("Contornos de Canny", WINDOW_AUTOSIZE);

  createTrackbar(TrackbarName,
                 "Contornos de Canny",
                 &top_slider,
                 top_slider_max,
                 on_trackbar_canny);

  on_trackbar_canny(top_slider, 0 );

  waitKey();

  imwrite("Pontilhismo_sem_o_uso_de_Canny.png", points);
  imwrite("Pontos_Canny.png", pointsCanny);
  imwrite("Imagem_mesclada.png", pointsFinal);
  imwrite("Contornos_Canny.png", border);

  return 0;
}


void on_trackbar_canny(int, void*){

  int raioCanny;
  int i, j;

  Vec3b valCanny;

  pointsCanny = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
  pointsFinal = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

  points.copyTo(pointsFinal);

  Canny(imageGray, border, top_slider, 3*top_slider);

  imshow("Contornos de Canny", border);

  for(i=0; i<height; i++){
    for(j=0; j<width; j++){
      if(border.at<uchar>(i,j) == 255){
        x = i;
        y = j;

        raioCanny = rand()%2 + 1;

        valCanny[0] = (int)(image.at<Vec3b>(x, y)[0]);
        valCanny[1] = (int)(image.at<Vec3b>(x, y)[1]);
        valCanny[2] = (int)(image.at<Vec3b>(x, y)[2]);

        circle(pointsCanny,
               Point(y, x),
               raioCanny,
               Scalar(valCanny[0], valCanny[1], valCanny[2]),
               -1,
               CV_AA);

        circle(pointsFinal,
               Point(y, x),
               raioCanny,
               Scalar(valCanny[0], valCanny[1], valCanny[2]),
               -1,
               CV_AA);
      }
    }
  }

  imshow("Pontos Canny.png", pointsCanny);
  imshow("Imagem mesclada.png", pointsFinal);

}
