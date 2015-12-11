/*
 * compile cmd:
 * g++ -std=c++11 `pkg-config --cflags opencv` classfiy.cc mxnet_predict-all.cc `pkg-config --libs opencv` -lopenblas -o classfiy
 */
#include "c_predict_api.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>
using namespace std;

float colors[3*224*224];
int loadrgb()
{
	ifstream out;
	out.open("./ReadImage/rgb.txt",ios::in);
	string temp;
	int i=0;
	while(getline(out,temp)){
		//cout << atof(temp.c_str())<< " ";
		colors[i]=atof(temp.c_str());
		i++;
	}
	//cout << i << endl;
	return 0;
}
//get the top N result
void top5(float* a){
	sort(a,a+1000);
	cout << "top1:" << a[999] << endl;
	cout << "top2:" << a[998] << endl;
	cout << "top3:" << a[997] << endl;
	cout << "top4:" << a[996] << endl;
	cout << "top5:" << a[995] << endl;
}



int main(){


	//read symbol
	ifstream ifile1("./raw/symbol.json");
	//ifstream ifile1("./raw/Inception_BN-symbol.json");
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
	//ifstream ifile2("./raw/Inception_BN-0039.params");
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
	//cout << "create_result = " << create_result << endl;
	if (create_result!=0)
		cout << MXGetLastError() << endl;

	//prepare the image data
	int x=loadrgb();
	const char* key="data";
	mx_uint imagearraysize=3*224*224;
	int setInput_result = MXPredSetInput(handle,key,colors,imagearraysize);
//	cout << "setinput_result: " << setInput_result << endl;
	if (setInput_result!=0)
			cout << MXGetLastError() << endl;

	//begin the forward
	int forward_result=MXPredForward(handle);
	//cout << "forward_result: "<<forward_result << endl;

	//get the output
	mx_uint *shape=0;
	mx_uint shape_len;
	int getoutputshape_result = MXPredGetOutputShape(handle,0,&shape,&shape_len);
	//cout << "getoutputshape_result: " << getoutputshape_result << endl;

	//get output
	size_t size=1;
	mx_uint m;
	for (m=0;m<shape_len;++m) size*=shape[m];
	float output[size];
	int getoutput_result = MXPredGetOutput(handle,0,output,size);
	//cout << "getoutput_result: " << getoutput_result << endl;
	cout << "+++++++  test output   +++++++" << endl;
	top5(output);
	//free the predictor
	int free_result = MXPredFree(handle);

	return 0;
}
