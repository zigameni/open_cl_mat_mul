// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS


#include <iostream>
#include <CL/cl.hpp>
#include <fstream>

int main()
{
    /* Part 1 */
    //std::cout << "Hello World!\n";

    //cl::Platform platform;


    /* Part 2 */
    
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    //_ASSERT(platforms.size() > 0);

    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    //_ASSERT(devices.size() > 0);
    
    auto device = devices.front();
    auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
    auto version = device.getInfo<CL_DEVICE_VERSION>();
    std::cout << vendor<<std::endl;
    std::cout << version << std::endl;
  

    /* Part 3, Kernels*/
    /* Simplest kernel you could have, it does nothing
    
    __kernel void functionName(__global int* inData) {

        return void;
    }
    */
    std::ifstream hello("hello.cl");
    std::string src(std::istreambuf_iterator<char>(hello), (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(device);
    cl::Program program(context, sources);

    auto err = program.build("-cl-std-CL1.2");

    char buf[16];
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));

    cl::Kernel kernel(program, "hello", &err);

    kernel.setArg(0, memBuf);

    cl::CommandQueue queue(context, device);
    queue.enqueueTask(kernel);

    queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);

    std::cout << buf;
    std::cin.get();




    return 0;
}

