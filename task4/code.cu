
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <cub/cub.cuh>

#define CUDACHECK(name) if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error(name);

__global__ void edge_interpolation(double* A, double* Anew, double step)
{
    int size = blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    //inits edge values
    A[index] = A[0] + index * step;
    A[index * size] = A[0] + index * step;
    A[size - 1 + size * index] = A[size - 1] + index * step;
    A[size * (size - 1) + index] = A[size * (size - 1)] + index * step;

    Anew[index] = Anew[0] + index * step;
    Anew[index * size] = Anew[0] + index * step;
    Anew[size - 1 + size * index] = Anew[size - 1] + index * step;
    Anew[size * (size - 1) + index] = Anew[size * (size - 1)] + index * step;
}

__global__ void interpolate(double* A, double* Anew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * blockDim.x + x;
    if (index > blockDim.x && index < blockDim.x * (blockDim.x - 1) - 1) {
        int residual = index % blockDim.x;
        if (residual == 0 || residual == blockDim.x - 1) {
            return;
        }
    }
    else {
        return;
    }

    //average between neighbours
    Anew[index] = 0.25 * (A[index + 1] + A[index - 1] + A[index + blockDim.x] + A[index - blockDim.x]);
}

__global__ void abs_diff(double* A, double* Anew) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * blockDim.x + x;

    A[index] = A[index] - Anew[index];
    A[index] = A[index] < 0 ? (A[index] * (-1)) : A[index];
}

int main(int argc, char* argv[])
{
    //reads command prompt arguments: ./task4.out [max_aaccuracy] [size] [max_iterations]
    double max_accuracy = std::stod(argv[1]);
    int size = std::stoi(argv[2]);
    int matrixSize = size * size;
    int max_iterations = std::stoi(argv[3]);

    //allocate matrixes and set start conditions (angle values)
    double* A = new double[matrixSize];
    double* Anew = new double[matrixSize];
    memset(A, 0, matrixSize * sizeof(double));
    memset(Anew, 0, matrixSize * sizeof(double));

    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[size * size - 1] = 30.0;
    A[size * (size - 1)] = 20.0;

    Anew[0] = 10.0;
    Anew[size - 1] = 20.0;
    Anew[size * size - 1] = 30.0;
    Anew[size * (size - 1)] = 20.0;

    //allocates data on GPU
    double* buff; //buffer for reduciton
    double* dev_A; //GPU copy of matrix A
    double* dev_Anew; //GPU copy of matrix Anew
    cudaMalloc((void**)&buff, matrixSize * sizeof(double));
    CUDACHECK("alloc buff");
    cudaMalloc((void**)&dev_A, matrixSize * sizeof(double));
    CUDACHECK("alloc dev_A");
    cudaMalloc((void**)&dev_Anew, matrixSize * sizeof(double));
    CUDACHECK("alloc dev_Anew");

    //copies values in matrixes 'A' and 'Anew' from CPU to GPU
    cudaMemcpy(dev_A, A, matrixSize * sizeof(double), cudaMemcpyHostToDevice);
    CUDACHECK("copy from A to dev_A");
    cudaMemcpy(dev_Anew, Anew, matrixSize * sizeof(double), cudaMemcpyHostToDevice);
    CUDACHECK("copy from Anew to dev_Anew");

    double step = 10.0 / (size - 1);
    edge_interpolation<<<1, size>>>(dev_A, dev_Anew, step);

    //allocates buffer 'd_out' to contain max('abs_diff' function result)
    double* d_out;
    cudaMalloc((void**)&d_out, sizeof(double));
    CUDACHECK("alloc d_out");

    //allocates temporary storage for Max operation and sets temp_storage bytes
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, matrixSize);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    size_t threads = (size < 1024) ? size : 1024;
    unsigned int blocks = size / threads;

	dim3 blockDim(threads / 32, threads / 32);
    dim3 gridDim(blocks * 32, blocks * 32);

    double accuracy = max_accuracy + 1.0;
    int num_of_iterations = 0;
    while (num_of_iterations < max_iterations && accuracy > max_accuracy) {

        interpolate<<<gridDim, blockDim>>>(dev_A, dev_Anew);

        //updates accuracy 1/100 times of main cycle iterations
        if (num_of_iterations % 100 == 0 || num_of_iterations + 1 == max_iterations) {

            //fills 'buff' with values from 'dev_A'
            cudaMemcpy(buff, dev_A, matrixSize * sizeof(double), cudaMemcpyDeviceToDevice);
            CUDACHECK("update dev_A");

            abs_diff<<<gridDim, blockDim>>>(buff, dev_Anew);

            //max reduction
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, matrixSize);

            cudaMemcpy(&accuracy, d_out, sizeof(double), cudaMemcpyDeviceToHost);
            CUDACHECK("copy to accuracy");
        }

        ++num_of_iterations;
        std::swap(dev_A, dev_Anew);
    }

    printf("Iterations: %d\nAccuracy: %lf\n", num_of_iterations, accuracy);

    //free memory section
    //GPU free
    cudaFree(dev_A);
    CUDACHECK("free dev_A");
    cudaFree(dev_Anew);
    CUDACHECK("free dev_Anew");
    cudaFree(buff);
    CUDACHECK("free buff");

    //CPU free
    free(A);
    free(Anew);

    return 0;
}
