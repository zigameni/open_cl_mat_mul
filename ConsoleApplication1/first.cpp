#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <CL/cl.hpp>

#include <vector>
#define ROWS (4)
#define COLUMNS (4)




void testing_opencl() {
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";


    cl::Context context({ default_device });

    cl::Program::Sources sources;

    // kernel calculates for each element C=A+B
    std::string kernel_code =
        "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
        "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
        "   }                                                                               ";

    /*
    std::string kernel_code =
        "void kernel matrix_mul(const int M, const int K, const int N, "
        "__global const float* a,   "
        "__global const float* b,   "
        "__global float *c)         "
        "{"
        "   int i = get_global_id(0);  "
        "   int j = get_global_id(1);  "
        "   float s = 0;               "
        "   if(i<M && j < N){"
        "       for (int k < 0; k < K; ++k)        "
        "           s += a[i * K + k] * b[k * N + j];  "
        "   }"
        "   c[i*N +j] = s";
    "}";
    */
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    cl::Program program(context, sources);
    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }


    // create buffers on the device
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

    int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context, default_device);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);



    //run the kernel
    //cl::KernelFunctor simple_add(cl::Kernel(program, "simple_add"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
    //simple_add(buffer_A, buffer_B, buffer_C);

    //alternative way to run the kernel
    cl::Kernel kernel_add = cl::Kernel(program, "simple_add");
    kernel_add.setArg(0, buffer_A);
    kernel_add.setArg(1, buffer_B);
    kernel_add.setArg(2, buffer_C);
    queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(10), cl::NullRange);
    queue.finish();

    int C[10];
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

    std::cout << " result: \n";
    for (int i = 0;i < 10;i++) {
        std::cout << C[i] << " ";
    }



}

void simple_matrix_multiplication() {
    int A;

    // Fill vectors X and Y with random float values

    float* h_x = new float[ROWS * COLUMNS];
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLUMNS; ++j) {
            h_x[j + i * COLUMNS] = rand() / (float)RAND_MAX;;
        }
    }
    float* h_y = new float[ROWS * COLUMNS];
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLUMNS; ++j) {
            h_y[j + i * COLUMNS] = rand() / (float)RAND_MAX;;
        }
    }
    float* h_s = new float[ROWS * COLUMNS];
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLUMNS; ++j) {
            h_s[j + i * COLUMNS] = 0.0;
        }
    }

    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    // Get all platforms (drivers)

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);


    if (all_platforms.size() == 0) { // Check for issues
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    // Get default device of the default platform

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);

    if (all_devices.size() == 0) { // Check for issues
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    // Create an OpenCL context

    cl::Context context({ default_device });

    cl::Program::Sources sources;

    std::string kernel_code =
        "void kernel simple_mul(       "
        "__global float* X,                     "
        "__global float* Y,                     "
        "__global float* S,                     "
        "__global int* A){                      "
        "    S[get_global_id(0)] = X[get_global_id(0)] * Y[get_global_id(0)]; "
        "} ";

    sources.push_back({ kernel_code.c_str(),kernel_code.length() });
    cl::Program program(context, sources);

    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        getchar();
        exit(1);
    }

    // create buffers on the device
    cl::Buffer buffer_X(context, CL_MEM_READ_WRITE, sizeof(float) * ROWS * COLUMNS);
    cl::Buffer buffer_Y(context, CL_MEM_READ_WRITE, sizeof(float) * ROWS * COLUMNS);
    cl::Buffer buffer_S(context, CL_MEM_READ_WRITE, sizeof(float) * ROWS * COLUMNS);
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int));

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context, default_device);

    //write arrays A and B to the device
    queue.enqueueWriteBuffer(buffer_X, CL_TRUE, 0, sizeof(float) * ROWS * COLUMNS, &h_x[0]);
    queue.enqueueWriteBuffer(buffer_Y, CL_TRUE, 0, sizeof(float) * ROWS * COLUMNS, &h_y[0]);
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int), &A);

    //StartCounter();
    //run the kernel
    cl::Kernel kernel_add = cl::Kernel(program, "simple_mul");
    kernel_add.setArg(0, buffer_X);
    kernel_add.setArg(1, buffer_Y);
    kernel_add.setArg(2, buffer_S);
    kernel_add.setArg(3, buffer_A);

    cl::NDRange global(ROWS * COLUMNS);
    queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, global, cl::NullRange);
    queue.finish();

    // std::cout << "Kernel execution time: " << GetCounter() << "ms \n";

    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_S, CL_TRUE, 0, sizeof(float) * ROWS * COLUMNS, &h_s[0]);



    /*Print vectors
    std::cout << "\nMatrix #1: \n";
    for (int i = 0; i<ROWS*COLUMNS; i++){
            std::cout << "" << h_x[i] << "\t ";
    }

    std::cout << "\n\nMatrix #2: \n";
    for (int i = 0; i<ROWS*COLUMNS; i++){
            std::cout << "" << h_y[i] << "\t ";

    }
    */
    std::cout << "\n\nResult: \n";
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            std::cout << "" << h_s[i * j] << "\t ";
        }
        std::cout << std::endl;
    }
    getchar();
}



int main() {
    testing_opencl();
    simple_matrix_multiplication();
    return 0;
}