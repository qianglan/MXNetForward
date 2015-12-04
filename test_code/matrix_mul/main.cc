#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// name of the file which contais the clkernel function
#define PROGRAM_FILE "matrix_multiply.cl"
// name of the clkernel function
#define KERNEL_FUNC "multiply_matrix"
// number of lines of the first matrix:M
#define LM1 2916
// number of columns of the first matrix and number of
// the second matrix:K
#define CM 363
// number of columns of the second matrix:N
#define CM2 64

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
      printf("GPU not found , using CPU\n");
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

/**
*  Construct a matrix from a vector.
*  Receives the vector, its length, and the desired column length.
*  The function works well if the relation between the vector length
* and the desired columns length results in an integer number.
*  Returns a pointer to the first element of the matrix
*/
float** vector_to_matrix(float* v, unsigned long int vlen, unsigned long int cv) {
   unsigned long int mlines = vlen/cv;
   unsigned long int cli,clj,k=0;
   float** r=(float**)malloc(sizeof(float*)*mlines);
   for(cli=0;cli<mlines;++cli)
      r[cli]=(float*)malloc(sizeof(float)*cv);

   for(cli=0;cli<mlines;++cli)
      for(clj=0;clj<cv;++clj)
         r[cli][clj]=v[k++];

   return r;
}

/**
*  Construct a vector from a matrix
*  Receives the matrix, its number of lines, its number of columns and optionally the
* reference for a variable which will keep the result vector length.
*  Returns a pointer to the initial position of the vector
*/
float* matrix_to_vector(float** m, unsigned long int l, unsigned long int c, unsigned long int *vlen) {
   unsigned long int cli,clj,k=0;
   float *v = (float*)malloc(sizeof(float)*l*c);
   for(cli=0;cli<l;++cli)
      for(clj=0;clj<c;++clj)
         v[k++]=m[cli][clj];
   if(vlen != NULL)
      *vlen = l*c;
   return v;
}

/**
*  Auxiliary function used to print a vector.
*  Receives the reference to the vector and its length
*/
void print_vector(float *v, unsigned long int len) {
   unsigned long int cli;
   for(cli=0;cli<len;++cli)
      printf("%.2f ",v[cli]);
   printf("\n");
}

// MAIN FUNCTION
int main() {
   // Declaration of OpenCL structures
   cl_device_id cldevice;
   cl_context clcontext;
   cl_program clprogram;
   cl_kernel clkernel;
   cl_command_queue clqueue;
   cl_int cli, clj, clerr;
   // variables to the number of threads in one block
   // and total numbers of threads, respectively
   unsigned long int local_size, global_size;

   // The data matrices
   float **m1=(float**)malloc(sizeof(float*)*LM1);
   float **m2=(float**)malloc(sizeof(float*)*CM);
   float **res=(float**)malloc(sizeof(float*)*LM1);
   // the vector which will be send to the devices
   cl_mem d_m1, d_m2, d_res;

   // allocates and initializes the data matrices
   for(cli=0;cli<LM1;++cli) {
      m1[cli]=(float*)malloc(sizeof(float)*CM);
      for(clj=0;clj<CM;++clj)
         m1[cli][clj] = 1.0f;
   }

   for(cli=0;cli<CM;++cli) {
      m2[cli]=(float*)malloc(sizeof(float)*CM2);
      for(clj=0;clj<CM2;++clj)
         m2[cli][clj] = 1.0f;
   }

   for(cli=0;cli<LM1;++cli) {
      res[cli]=(float*)malloc(sizeof(float)*CM2);
      for(clj=0;clj<CM2;++clj)
         res[cli][clj] = 0.0f;
   }

   // Create a cldevice and the clcontext
   cldevice = create_device();
   clcontext = clCreateContext(NULL, 1, &cldevice, NULL, NULL, &clerr);
   if(clerr < 0) {
      perror("Couldn't create a clcontext");
      exit(1);
   }

   // creates a clprogram and build it
   clprogram = build_program(clcontext, cldevice, PROGRAM_FILE);

   // Create the data buffers size
   unsigned long int m1size = sizeof(float)*LM1*CM;
   unsigned long int m2size = sizeof(float)*CM*CM2;
   unsigned long int res_size = sizeof(float)*LM1*CM2;

   float *vm1, *vm2, *vres;
   // Make a vector with each of the allocated matrices
   vm1 = matrix_to_vector(m1,(int)LM1,(int)CM, NULL);
   vm2 = matrix_to_vector(m2,(int)CM,(int)CM2, NULL);
   vres = matrix_to_vector(res,(int)LM1,(int)CM2,NULL);

   // defines the total number of threads
   global_size = LM1*CM2;
   // defines the number of threads in one block
   local_size = CM2;

   // create the data buffers to be sent to devices
   d_m1 = clCreateBuffer(clcontext, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, m1size, vm1, &clerr);
   d_m2 = clCreateBuffer(clcontext, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, m2size, vm2, &clerr);
   d_res = clCreateBuffer(clcontext, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, res_size, vres, &clerr);
   if(clerr < 0) {
      perror("Couldn't create a buffer");
      exit(1);
   };

   // Create a command clqueue
   clqueue = clCreateCommandQueue(clcontext, cldevice, 0, &clerr);
   if(clerr < 0) {
      perror("Couldn't create a command clqueue");
      exit(1);
   };

   // Create a clkernel
   clkernel = clCreateKernel(clprogram, KERNEL_FUNC, &clerr);
   if(clerr < 0) {
      perror("Couldn't create a clkernel");
      exit(1);
   };

   unsigned long int wm1 = CM;
   unsigned long int wm2 = CM2;
   // Sets the clkernel arguments
   clerr = clSetKernelArg(clkernel, 0, sizeof(cl_mem), &d_m1);
   clerr |= clSetKernelArg(clkernel, 1, sizeof(cl_mem), &d_m2);
   clerr |= clSetKernelArg(clkernel, 2, sizeof(cl_mem), &d_res);
   clerr |= clSetKernelArg(clkernel, 3, sizeof(unsigned long int),(void*) &wm1);
   clerr |= clSetKernelArg(clkernel, 4, sizeof(unsigned long int),(void*) &wm2);
   if(clerr < 0) {
      perror("Couldn't create a clkernel argument");
      exit(1);
   }

   // Enqueue the created clkernel
   clerr = clEnqueueNDRangeKernel(clqueue, clkernel, 1, NULL, &global_size,
         &local_size, 0, NULL, NULL);
   if(clerr < 0) {
      perror("Couldn't enqueue the clkernel PORRA");
      exit(1);
   }

   // Read the clkernel's output
   clerr = clEnqueueReadBuffer(clqueue, d_res, CL_TRUE, 0,
         res_size, vres, 0, NULL, NULL);
   if(clerr < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   printf("\n---finished---\n");
   /*
   // transforms the result to a matrix and print the values
   printf("====================================\n");
   res = vector_to_matrix(vres,(unsigned int)LM1*CM2,CM2);
   for(cli=0;cli<LM1;++cli) {
      for(clj=0;clj<CM2;++clj)
         printf("%.2f ", res[cli][clj]);
      printf("\n");
   }
   printf("====================================\n");
   */
   // Deallocating resources
   clReleaseKernel(clkernel);
   clReleaseMemObject(d_m1);
   clReleaseMemObject(d_m2);
   clReleaseMemObject(d_res);
   clReleaseCommandQueue(clqueue);
   clReleaseProgram(clprogram);
   clReleaseContext(clcontext);

   free(m1);
   free(m2);
   free(res);
   free(vm1);
   free(vm2);
   free(vres);
   return 0;
}
