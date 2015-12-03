#include <highgui.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace std;

float colors[3*224*224];
float mean_color=117.0f;

void readImage()
{
	cv::Mat inputImage = cv::imread("cd.bmp");
	int imageHeight = inputImage.rows;
	int imageWidth = inputImage.cols;
	cout << "Height: " << imageHeight << endl;
	cout << "Width: " << imageWidth << endl;
	uchar* inputImageData = inputImage.data;

	for (int i=0;i<imageHeight;i++){
		for(int j=0;j<imageWidth;j++){
			int index = 3*(i*imageWidth+j);
			int b = inputImageData[index];
			int g = inputImageData[index+1];
			int r = inputImageData[index+2];
      //cout << b << endl;
			colors[0*224*224+i*imageHeight+imageWidth]=(float)r-mean_color;
			colors[1*224*224+i*imageHeight+imageWidth]=(float)g-mean_color;
			colors[2*224*224+i*imageHeight+imageWidth]=(float)b-mean_color;
		}
	}
}

void save2file(){
  ofstream inFile;
  inFile.open("rgb.txt",ios::trunc);
  int i=0;
	//for (i=0;i<3*224*224;i++)
    //cout << colors[i] << " ";
  for (i=0;i<3*224*224;i++)
    inFile << colors[i] << "\n";
  inFile.close();
}

int main()
{
  readImage();
  save2file();
  return 0;
}
