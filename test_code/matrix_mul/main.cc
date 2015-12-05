#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "mygemm.h"
#include <iostream>
#include <math.h>
#include <ctime>
#include <unistd.h>
// name of the file which contais the clkernel function
#define PROGRAM_FILE "kernels_sgemm.cl"
// name of the clkernel function
#define KERNEL_FUNC "myGEMM"
using namespace std;

int M[8]={2916,625,144,144,144,144,144,1000};
int N[8]={64,192,96,160,144,160,256,1};
int K[8]={363,576,1728,864,1440,1296,1440,256};
int asize=0;
int bsize=0;
int csize=0;
// Declaration of OpenCL structures (global)
cl_device_id cldevice;
cl_context clcontext;
cl_program clprogram;
cl_kernel clkernel;
cl_command_queue clqueue;
cl_int cli, clj, clerr;

/**
*  Find a GPU or CPU (cldevice) which is available for the host returning
* the created cldevice
*/
cl_device_id create_device() {
   cl_platform_id platform;
   cl_device_id dev;
   cl_int clerr;
   // Identify a platform
   clerr = clGetPlatformIDs(1, &platform, NULL);
   if(clerr < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   }
   // Try to access a GPU
   clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(clerr == CL_DEVICE_NOT_FOUND) {
      //printf("GPU not found , using CPU\n");
      // if can't. Try to access a CPU
      clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   // If there's an error finish the clprogram
   if(clerr < 0) {
      perror("Couldn't access any devices");
      exit(1);
   }
   // return the cldevice id
   return dev;
}


/**
*  Create a clprogram (clkernel function) from a file, returning it
* compiled to the caller.
*  Receives a opencl clcontext, a cldevice ID and the name of the file
* which contains the clprogram.
*/
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program clprogram;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int clerr;

   // Read clprogram file and place its content into a buffer
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the clprogram file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   // gets the clprogram size
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   // sets the end of the buffer
   program_buffer[program_size] = '\0';
   // reads the file content and close it
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   // Create clprogram from file
   clprogram = clCreateProgramWithSource(ctx, 1,
      (const char**)&program_buffer, &program_size, &clerr);
   if(clerr < 0) {
      perror("Couldn't create the clprogram");
      exit(1);
   }
   // deallocate the clprogram buffer
   free(program_buffer);

   // Build the read clprogram
   clerr = clBuildProgram(clprogram, 0, NULL, NULL, NULL, NULL);
   if(clerr < 0) {

      // Find size of log and print to std output
      clGetProgramBuildInfo(clprogram, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(clprogram, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      // prints the log with the error informations
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return clprogram;
}


void initOpenCL(){
	// Create a cldevice and the clcontext
	cldevice = create_device();
	clcontext = clCreateContext(NULL, 1, &cldevice, NULL, NULL, &clerr);
	if(clerr < 0) {
		 perror("Couldn't create a clcontext");
		 exit(1);
	}
	// creates a clprogram and build it
	clprogram = build_program(clcontext, cldevice, PROGRAM_FILE);
	// Create a command clqueue
	clqueue = clCreateCommandQueue(clcontext, cldevice, 0, &clerr);
	if(clerr < 0) {

		 perror("Couldn't create a command clqueue");
		 exit(1);
	}

	// Create a clkernel
	clkernel = clCreateKernel(clprogram, KERNEL_FUNC, &clerr);
	if(clerr < 0) {
		 perror("Couldn't create a clkernel ^^^^");
		 //LOG(INFO) << "the error num is clerr = " << clerr;
		 exit(1);
	}
}



void initABC(float* a,float* b,int t){
  int i=0;
  sleep(1);
  srand((unsigned)time(0));
  for (i=0;i<asize;i++)
    a[i]=rand()/100000000.0;
  for (i=0;i<bsize;i++)
    b[i]=rand()/100000000.0;
  //for (i=0;i<M[t]*N[t];i++)
    //c[i]=rand()/1000.0;
  cout << "init the matrix A ,B Bsuccess.." << "A[" << 7 << "]= " <<a[7] <<"  B[" << 7 << "]= " <<b[7] << endl;
}

int main(){
  int call = 0;
  initOpenCL();
  cout << "====Begin the test: ========================================" << endl;
  for (call=0;call<8;call++){
    cout << "test " << call << ":  " << endl;
    asize=M[call]*K[call];
    bsize=N[call]*K[call];
    csize=M[call]*N[call];
    float* A = new float[asize];
    float* B = new float[bsize];
    float* C = new float[csize];
    float* clc = new float[csize];

    initABC(A,B,call);
    // use mygemm to do
    my_sgemm(false,false,M[call],N[call],K[call],1,A,0,B,0,0.0,C,0);
    //use cl to do it
    // variables to the number of threads in one block
    // and total numbers of threads, respectively

    unsigned long int local_size, global_size;
    // the vector which will be send to the devices
    cl_mem d_m1, d_m2, d_res;
    // Create the data buffers size
    unsigned long int m1size = sizeof(float)*M[call]*K[call];
    unsigned long int m2size = sizeof(float)*N[call]*K[call];
    unsigned long int res_size = sizeof(float)*M[call]*N[call];
    // defines the total number of threads
    global_size = M[call]*N[call];
    // defines the number of threads in one block
    local_size = M[call];
    float* temp_a = (float*) A;
    float* temp_b = (float*) B;
    float* temp_c = (float*) clc;

    // create the data buffers to be sent to devices
    d_m1 = clCreateBuffer(clcontext, CL_MEM_READ_ONLY |
          CL_MEM_COPY_HOST_PTR, m1size, temp_a, &clerr);
    d_m2 = clCreateBuffer(clcontext, CL_MEM_READ_ONLY |
          CL_MEM_COPY_HOST_PTR, m2size, temp_b, &clerr);
    d_res = clCreateBuffer(clcontext, CL_MEM_READ_WRITE |
          CL_MEM_COPY_HOST_PTR, res_size, temp_c, &clerr);
    if(clerr < 0) {
       perror("Couldn't create a buffer");
       exit(1);
    }

    const int clM = M[call];
    const int clN = N[call];
    const int clK = K[call];
    int ta=0;

    clerr = clSetKernelArg(clkernel, 0, sizeof(cl_int),(void*) &clM);
    clerr |= clSetKernelArg(clkernel, 1, sizeof(cl_int),(void*) &clN);
    clerr |= clSetKernelArg(clkernel, 2, sizeof(cl_int),(void*) &clK);
    clerr |= clSetKernelArg(clkernel, 3, sizeof(cl_mem), &d_m1);
    clerr |= clSetKernelArg(clkernel, 4, sizeof(cl_mem), &d_m2);
    clerr |= clSetKernelArg(clkernel, 5, sizeof(cl_mem), &d_res);
    clerr |= clSetKernelArg(clkernel, 6, sizeof(cl_int),(void*) &ta);
    if(clerr < 0) {
       perror("Couldn't create a clkernel argument");
       exit(1);
    }
    // Enqueue the created clkernel
    clerr = clEnqueueNDRangeKernel(clqueue, clkernel, 1, NULL, &global_size,
          &local_size, 0, NULL, NULL);
    if(clerr < 0) {
       perror("Couldn't enqueue the clkernel PORRA");
       //LOG(INFO) << "the error num is clerr = " << clerr;
       exit(1);
    }

    // Read the clkernel's output
    clerr = clEnqueueReadBuffer(clqueue, d_res, CL_TRUE, 0,
          res_size, clc, 0, NULL, NULL);
    if(clerr < 0) {
       perror("Couldn't read the buffer");
       exit(1);
    }

    clReleaseMemObject(d_m1);
    clReleaseMemObject(d_m2);
    clReleaseMemObject(d_res);

    //check the result:
    int tt;
    for (tt=0;tt<M[call]*N[call];tt=tt+1){
      if (fabs(clc[tt]-C[tt])<10e-4)
        continue;
      else
        break;
    }

    if (tt!=M[call]*N[call]){
      cout  << "OpenCL get the wrong answer!!!  TT = " << tt << "  in C:" << C[tt] << "  in clc:" << clc[tt] << endl;
    }
    else
      cout << "test " << call << " passed!" << endl;

    delete A;
    delete B;
    delete C;
    delete clc;

  }
  cout << "====test   end: ==========================================" << endl;
  clReleaseKernel(clkernel);
  clReleaseCommandQueue(clqueue);
  clReleaseProgram(clprogram);
  clReleaseContext(clcontext);

  return 0;
}
