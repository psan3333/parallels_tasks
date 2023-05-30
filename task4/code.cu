
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <cub/cub.cuh>

#define CUDACHECK(name) if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error(name);

__global__ void interpolate(double* A, double* Anew, int size)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i * size + j > size * size) return;
	
	if(!((j == 0 || i == 0 || j == size - 1 || i == size - 1)))
	{
		Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] +
							A[(i + 1) * size + j] + A[i * size + j + 1]);		
	}
}

__global__ void abs_diff(double* A, double* Anew, double* buff, int size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > size * size) return;

	buff[idx] = std::abs(Anew[idx] - A[idx]);
}

int main(int argc, char* argv[])
{
    //reads command prompt arguments: ./task4.out [max_aaccuracy] [size] [max_iterations]
    double max_accuracy = std::stod(argv[1]);
    int size = std::stoi(argv[2]);
    int matrixSize = size * size;
    int max_iterations = std::stoi(argv[3]);

    //allocate matrixes and set start conditions (angle values)
    double* A;
    double* Anew;
    cudaMallocHost((void**)&A, matrixSize * sizeof(double));
    CUDACHECK("A host alloc")
    cudaMallocHost((void**)&Anew, matrixSize * sizeof(double));
    CUDACHECK("Anew host alloc")
        
    std::memset(A, 0, matrixSize * sizeof(double));
    std::memset(Anew, 0, matrixSize * sizeof(double));

    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[size * size - 1] = 30.0;
    A[size * (size - 1)] = 20.0;

    Anew[0] = 10.0;
    Anew[size - 1] = 20.0;
    Anew[size * size - 1] = 30.0;
    Anew[size * (size - 1)] = 20.0;
    
    double step = 10.0 / (size - 1);
    for (int i = 1; i < size - 1; i++) {
        A[i] = A[0] + i * step;
        A[i * size] = A[0] + i * step;
        A[size - 1 + size * i] = A[size - 1] + i * step;
        A[size * (size - 1) + i] = A[size * (size - 1)] + i * step;

        Anew[i] = Anew[0] + i * step;
        Anew[i * size] = Anew[0] + i * step;
        Anew[size - 1 + size * i] = Anew[size - 1] + i * step;
        Anew[size * (size - 1) + i] = Anew[size * (size - 1)] + i * step;
    }

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

    //allocates buffer 'd_out' to contain max('abs_diff' function result)
    double* d_out;
    cudaMalloc((void**)&d_out, sizeof(double));
    CUDACHECK("alloc d_out");

    //allocates temporary storage for Max operation and sets temp_storage bytes
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, matrixSize);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    CUDACHECK("alloc d_temp_storage");
    
    bool isGraphCreated = false;
	cudaStream_t stream, memoryStream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&memoryStream);
	cudaGraph_t graph;
	cudaGraphExec_t instance;

	size_t threads = (size < 1024) ? size : 1024;
    unsigned int blocks = size / threads;

	dim3 blockDim(threads / 32, threads / 32);
    dim3 gridDim(blocks * 32, blocks * 32);

	int iter = 0;
	double accuracy = max_accuracy + 1.0;
	while(iter < max_iterations && accuracy > max_accuracy)
	{
		// Расчет матрицы
		if (isGraphCreated)
		{
			cudaGraphLaunch(instance, stream);
			
			cudaMemcpyAsync(&accuracy, d_out, sizeof(double), cudaMemcpyDeviceToHost, stream);

			cudaStreamSynchronize(stream);

			iter += 100;
		}
		else
		{
			cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
			for(size_t i = 0; i < 50; i++)
			{
				interpolate<<<gridDim, blockDim, 0, stream>>>(dev_A, dev_Anew, size);
				interpolate<<<gridDim, blockDim, 0, stream>>>(dev_Anew, dev_A, size);
			}
			// Расчитываем ошибку каждую сотую итерацию
			abs_diff<<<threads * blocks * blocks, threads, 0, stream>>>(dev_A, dev_Anew, buff, size);
			cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, matrixSize, stream);
	
			cudaStreamEndCapture(stream, &graph);
			cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
			isGraphCreated = true;
  		}
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
	cudaFree(d_temp_storage);
	CUDACHECK("free d_temp_storage");
	cudaFreeHost(A);
	CUDACHECK("free A");
	cudaFreeHost(Anew);
	CUDACHECK("free Anew");
	cudaFree(d_out);
	CUDACHECK("free d_out");
	cudaStreamDestroy(stream);
	CUDACHECK("destroy stream");
	cudaGraphDestroy(graph);
	CUDACHECK("destroy graph");


    return 0;
}
