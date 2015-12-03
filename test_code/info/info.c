#include <iostream>
#include <CL/cl.h>
using namespace std;

const char* GetDeviceType(cl_device_type it)
{
    if(it == CL_DEVICE_TYPE_CPU)
        return "CPU";
    else if(it== CL_DEVICE_TYPE_GPU)
        return "GPU";
    else if(it==CL_DEVICE_TYPE_ACCELERATOR)
        return "ACCELERATOR";
    else
        return "DEFAULT";

}

int main()
{
    char dname[512];
    cl_uint num_platform;
    cl_device_id devices[20];
    cl_platform_id platform_id = NULL;
    cl_device_type int_type;
    cl_uint num_devices;
    cl_ulong long_entries;

    cl_int err;

    //获取系统上可用的计算平台，可以理解为初始化
     clGetPlatformIDs(0,NULL,&num_platform);
     cout<<"CL_PLATFORM_NUM:"<<num_platform<<endl;

    //我的本本只有一个计算平台。如果有GPU的话可以选2
     err = clGetPlatformIDs(1,&platform_id,NULL);
    if(err!=CL_SUCCESS)
    {
        cout<<"clGetPlatform NUMerror"<<endl;
        return 0;
    }

    //获取可用计算平台的名称
    clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,512,dname,NULL);
    cout<<"CL_PLATFORM_NAME:"<<dname<<endl;

    //获取可用计算平台的版本号,即OpenCL的版本号
    clGetPlatformInfo(platform_id,CL_PLATFORM_VERSION,512,dname,NULL);
    cout<<"CL_PLATFORM_VERSION:"<<dname<<endl;

    //获取可用计算平台的设备数目
    clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_ALL,20,devices,&num_devices);
    cout<<"Device  num:"<<num_devices<<endl;

    clGetDeviceInfo(devices[0],CL_DEVICE_NAME,512,dname,NULL);
    cout<<"Device 1 Name:"<<dname<<endl;

    clGetDeviceInfo(devices[0],CL_DEVICE_TYPE,sizeof(cl_device_type),&int_type,NULL);
    cout<<"Device Type:"<<GetDeviceType(int_type)<<endl;

    return 0;
    return 0;
}
