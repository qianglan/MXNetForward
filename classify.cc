/*
 * compile cmd:
 * g++ -std=c++11 `pkg-config --cflags opencv` classfiy.cc mxnet_predict-all.cc `pkg-config --libs opencv` -lopenblas -o classfiy
 */


#include "c_predict_api.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <highgui.h>
using namespace std;

const float mean_color = 117.0;
float colors[3*224*224];

int readImage()
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
			colors[0*224*224+i*imageHeight+imageWidth]=(float)r-mean_color;
			colors[1*224*224+i*imageHeight+imageWidth]=(float)g-mean_color;
			colors[2*224*224+i*imageHeight+imageWidth]=(float)b-mean_color;
		}
	}
	return 0;
}



int main(){
	/*
	 * init the predictor
	 */
	//read symbol
	ifstream ifile1("./raw/symbol.json");
	ostringstream buf1;
	char ch1;
	while (buf1&&ifile1.get(ch1))
		buf1.put(ch1);
	string res1;
	res1=buf1.str();
	//cout<< res << endl;
	const char* symbol_json=res1.c_str();
	//cout << symbol_json << endl;
	ifile1.close();

	//read params
	ifstream ifile2("./raw/params");
	ostringstream buf2;
	char ch2;
	int num=0;
	while (buf2&&ifile2.get(ch2)){
		buf2.put(ch2);
		num++;
	}
	string res2;
	res2=buf2.str();
	const char* params=res2.c_str();
	ifile2.close();

	// other params needed for the predictor,set the predictor
	size_t param_buf_size=num;
	int dev_type=1;
	int dev_id=0;
	mx_uint num_input_nodes=1;
	const char** input_keys;
	input_keys=(const char**)malloc(sizeof(char**));
	*input_keys ="data";
	const mx_uint input_shape_indptr[2]={0,4};
	const mx_uint input_shape_data[4]={1,3,224,224};
	PredictorHandle handle = 0;
	int create_result =MXPredCreate(symbol_json,params,param_buf_size,dev_type,dev_id,num_input_nodes,input_keys,input_shape_indptr,input_shape_data,&handle);
	cout << "create_result = " << create_result << endl;
	if (create_result!=0)
		cout << MXGetLastError() << endl;

	//prepare the image data
	int x=readImage();
	const char* key="data";
	mx_uint imagearraysize=3*224*224;
	int setInput_result = MXPredSetInput(handle,key,colors,imagearraysize);
	cout << "setinput_result: " << setInput_result << endl;
	if (setInput_result!=0)
			cout << MXGetLastError() << endl;

	//begin the forward
	int forward_result=MXPredForward(handle);
	cout << "forward_result: "<<forward_result << endl;

	//get the output
	mx_uint *shape=0;
	mx_uint shape_len;
	int getoutputshape_result = MXPredGetOutputShape(handle,0,&shape,&shape_len);
	cout << "getoutputshape_result: " << getoutputshape_result << endl;

	//get output
	size_t size=1;
	mx_uint m;
	for (m=0;m<shape_len;++m) size*=shape[m];
	float output[size];
	int getoutput_result = MXPredGetOutput(handle,0,output,size);
	cout << "getoutput_result: " << getoutput_result << endl;
	//for (m=0;m<size;m++)
	//	cout << "output[" << m << "]: " << output[m] << endl;

	//free the predictor
	int free_result = MXPredFree(handle);
	return 0;
}




