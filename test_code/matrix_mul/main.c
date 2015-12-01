/*
*compile cmd:gcc main.c -lrt -lOpenCL -o mul
for now , MATRIX_A =4 is right ,other value is wrong
*/


#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "matvec.cl"
#define KERNEL_FUNC "matvec_mult"
#define MATRIX_A_SIZE  4

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif





int main(void)
{

  // for time
  time_t t_start,t_end;

 //主机/设备的数据结构
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_int i, err;

//程序/内核数据结构
  cl_program program;
  FILE *program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;
  cl_kernel kernel;

//数据和缓冲区
  float mat[MATRIX_A_SIZE*MATRIX_A_SIZE], vec[MATRIX_A_SIZE], result[MATRIX_A_SIZE];
  float correct[MATRIX_A_SIZE] = {0.0f};
  cl_mem mat_buff, vec_buff, res_buff;
  size_t work_units_per_kernel;

//数据初始化
  //printf("%s\n", "mat:");
  for(i=0; i<MATRIX_A_SIZE*MATRIX_A_SIZE; i++) {
    mat[i] = i * 2.0f;
    //printf("%f  ",mat[i]);
  }
  //printf("\n\n");

  //printf("%s\n","vec:" );
  for(i=0; i<MATRIX_A_SIZE; i++) {
    vec[i] = i * 3.0f;
    //printf("%f  ",vec[i]);
  }
  //printf("\n\n");

  int j=0;
  t_start=time(NULL);
  for (i=0;i<MATRIX_A_SIZE;i++){
    for (j=0;j<MATRIX_A_SIZE;j++){
      correct[i] += mat[i*MATRIX_A_SIZE+j] * vec[j];
    }
  }
  t_end=time(NULL);
  printf("the time of pure cpu is: %f \n", difftime(t_end,t_start));

/*
  //printf("%s\n","correct:" );
  for(i=0;i<MATRIX_A_SIZE;i++){
    printf("%f\n",correct[i] );
  }
  //printf("\n\n");
*/
  //检查支持 OpenCL 的平台
  err = clGetPlatformIDs(1, &platform, NULL);
  if(err < 0) {
    perror("找不到任何支持 OpenCL 的平台");
    exit(1);
  }

  //检查支持 OpenCL 的设备
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  if(err < 0) {
    perror("找不到任何支持 OpenCL 的设备");
    exit(1);
  }

  //创建上下文
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if(err < 0) {
    perror("创建上下文失败");
    exit(1);
  }

  //读取内核源码并放入缓冲区
  program_handle = fopen(PROGRAM_FILE, "r");
  if(program_handle == NULL) {
    perror("找不到内核文件");
    exit(1);
  }
  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char*)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  //创建内核程序
  program = clCreateProgramWithSource(context, 1,(const char**)&program_buffer, &program_size, &err);
  if(err < 0) {
    perror("创建内核程序失败");
    exit(1);
  }
  free(program_buffer);

//建立程序
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err < 0) {
  //查找日志的大小并打印输出
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
      0, NULL, &log_size);
    program_log = (char*) malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
    log_size + 1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    exit(1);
  }

  //创建内核
  kernel = clCreateKernel(program, KERNEL_FUNC, &err);
  if(err < 0) {
    perror("创建内核失败");
    exit(1);
  }

  //创建存放输入输出数据的缓冲区
  mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*MATRIX_A_SIZE*MATRIX_A_SIZE, mat, &err);
  if(err < 0) {
    perror("创建缓冲区失败");
    exit(1);
  }
  vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*MATRIX_A_SIZE, vec, NULL);
  res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,sizeof(float)*MATRIX_A_SIZE, NULL, NULL);

  //从缓冲区创建内核参数
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
  if(err < 0) {
    perror("设置内核参数失败");
    exit(1);
  }
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);

  //为设备创建命令队列
  queue = clCreateCommandQueue(context, device, 0, &err);
  if(err < 0) {
    perror("创建命令队列失败");
    exit(1);
  }

  t_start=time(NULL);
  //Enqueue the command queue to the device
  //4 work-units per kernel
  work_units_per_kernel = 4;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,NULL, 0, NULL, NULL);
  if(err < 0) {
    perror("Couldn't enqueue the kernel execution command");
    exit(1);
  }
  t_end=time(NULL);
  printf("the time of OpenCL cpu is: %f \n", difftime(t_end,t_start));
  //读取结果
  err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*MATRIX_A_SIZE,result, 0, NULL, NULL);
  if(err < 0) {
    perror("读取结果失败");
    exit(1);
  }

  //检查结果
  for (j=0;j<MATRIX_A_SIZE;j++)
  {
    //printf("%f  -  %f\n",result[j] ,correct[j]);


    if(result[j]==correct[j])
      continue;
    else
      break;

  }
  if(j==MATRIX_A_SIZE) {
    printf("矩阵矢量乘法成功\n");
  }
  else {
    printf("矩阵矢量乘法有问题\n");
    printf("j= %d\n", j);
  }

  //释放资源
  clReleaseMemObject(mat_buff);
  clReleaseMemObject(vec_buff);
  clReleaseMemObject(res_buff);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  return 0;
}
