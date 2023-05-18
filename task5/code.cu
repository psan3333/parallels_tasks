#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <cub/cub.cuh>
#include "mpi.h"

// error checks
#define CUDA_CHECK(name) if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error(name);
#define MPI_CHECK(code, name) if (code != MPI_SUCCESS) std::runtime_error(name);


__global__ void interpolate(double* A, double* Anew, size_t size, size_t size_per_one_gpu)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(!(j == 0 || i == 0 || j == size - 1 || i == size_per_one_gpu - 1))
		Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);	
}

__global__ void abs_diff(double* A, double* Anew, double* buff) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    buff[index] = std::abs(A[index] - Anew[index]);
}

int main(int argc, char* argv[])
{
    //reads command prompt arguments: ./task4.out [max_aaccuracy] [size] [max_iterations]
    double max_accuracy = std::stod(argv[1]);
    int size = std::stoi(argv[2]);
    int matrixSize = size * size;
    int max_iterations = std::stoi(argv[3]);

    cudaError_t cub_error;
    int rank, size_for_one_group, error_code;
    error_code = MPI_Init(&argc, &argv);
    MPI_CHECK(error_code, "mpi initialization");

    error_code = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_CHECK(error_code, "mpi communicator rank_init");
    error_code = MPI_Comm_size(MPI_COMM_WORLD, &size_for_one_group);
    MPI_CHECK(error_code, "mpi communicator size_init");

	cudaSetDevice(rank);
    CUDA_CHECK("cuda set device");

	if (rank != 0)
        cudaDeviceEnablePeerAccess(rank - 1, 0);
        CUDA_CHECK("enable peer access (rank != 0)");
    if (rank != size_for_one_group - 1)
        cudaDeviceEnablePeerAccess(rank + 1, 0);
        CUDA_CHECK("enable peer access (rank != size_for_one_group - 1)");

	// Размечаем границы между устройствами
	size_t area_for_one_process = size / size_for_one_group;
	size_t start_y_idx = area_for_one_process * rank;

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

    double step = 10.0 / (size - 1);
    for (int i = 0; i < size; i++) {
        A[i] = A[0] + i * step;
        A[i * size] = A[0] + i * step;
        A[size - 1 + size * i] = A[size - 1] + i * step;
        A[size * (size - 1) + i] = A[size * (size - 1)] + i * step;

        Anew[i] = Anew[0] + i * step;
        Anew[i * size] = Anew[0] + i * step;
        Anew[size - 1 + size * i] = Anew[size - 1] + i * step;
        Anew[size * (size - 1) + i] = Anew[size * (size - 1)] + i * step;
    }

    if (rank != 0 && rank != size_for_one_group - 1)
	{
		area_for_one_process += 2;
	}
	else 
	{
		area_for_one_process += 1;
	}

	size_t alloc_memsize = size * area_for_one_process;

    unsigned int threads_x = (size < 1024) ? size : 1024;
    unsigned int blocks_y = area_for_one_process;
    unsigned int blocks_x = size / threads_x;

    dim3 blockDim(threads_x, 1);
    dim3 gridDim(blocks_x, blocks_y);

    //allocates data on GPU
    double* buff; //buffer for reduciton
    double* dev_A; //GPU copy of matrix A
    double* dev_Anew; //GPU copy of matrix Anew
    cudaMalloc((void**)&buff, alloc_memsize * sizeof(double));
    CUDA_CHECK("alloc buff");
    cudaMalloc((void**)&dev_A, alloc_memsize * sizeof(double));
    CUDA_CHECK("alloc dev_A");
    cudaMalloc((void**)&dev_Anew, alloc_memsize * sizeof(double));
    CUDA_CHECK("alloc dev_Anew");

    //copies values in matrixes 'A' and 'Anew' from CPU to GPU
    size_t offset = (rank != 0) ? size : 0;
	cudaMemset(dev_A, 0, sizeof(double) * alloc_memsize);
    CUDA_CHECK("memset dev_A");
	cudaMemset(dev_Anew, 0, sizeof(double) * alloc_memsize);
    CUDA_CHECK("memset dev_Anew");
 	cudaMemcpy(dev_A, A + (start_y_idx * size) - offset, sizeof(double) * alloc_memsize, cudaMemcpyHostToDevice);
    CUDA_CHECK("memcpy from A to dev_A from start_y_idx coordinate with offset");
	cudaMemcpy(dev_Anew, matrixB + (start_y_idx * size) - offset, sizeof(double) * alloc_memsize, cudaMemcpyHostToDevice);
    CUDA_CHECK("memcpy from Anew to dev_Anew with from start_y_idx coordinate offset");

    // allocates buffer 'd_out' to contain max('abs_diff' function result)
    double* d_out;
    cudaMalloc((void**)&d_out, sizeof(double));
    CUDA_CHECK("alloc d_out");

    // allocates memory for temporary storage for Max operation and sets temp_storage_bytes with size of d_temp_storage in bytes
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, matrixSize);
    CUDA_CHECK("get temp_storage_bytes");
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CUDA_CHECK("temp storage memory allocation");

    // 
    double accuracy = max_accuracy + 1.0;
    int num_of_iterations = 0;
    cudaStream_t cuda_stream;
	cudaStreamCreate(&cuda_stream);
    CUDA_CHECK("stream creation");
    while (num_of_iterations < max_iterations && accuracy > max_accuracy) {

        interpolate<<<blockDim, gridDim, 0, cuda_stream>>>(dev_A, dev_Anew);

        // updates accuracy 1/100 times of main cycle iterations
        if (num_of_iterations % 100 == 0 || num_of_iterations + 1 == max_iterations) {

            abs_diff<<<blocks_x * blocks_y, threads_x, 0, cuda_stream>>>(dev_A, dev_Anew, buff);

            // max reduction
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, matrixSize);
            CUDA_CHECK("cub max reduction");

            cudaStreamSynchronize(cuda_stream);
            CUDA_CHECK("stream synchronization (inside error calculations)");

            MPI_Allreduce((void*)d_out, (void*)d_out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            cudaMemcpyAsync(&accuracy, d_out, sizeof(double), cudaMemcpyDeviceToHost);
            CUDA_CHECK("copy to accuracy");
        }

        cudaStreamSynchronize(cuda_stream);
        CUDA_CHECK("stream synchronization (main loop)");

        if (rank != 0)
		{
		    MPI_Sendrecv(
                dev_Anew + size + 1, 
                size - 2, 
                MPI_DOUBLE, 
                rank - 1, 
                0, 
			    dev_Anew + 1, 
                size - 2, 
                MPI_DOUBLE, 
                rank - 1, 
                0, 
                MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE
            );
		}
		// Обмен нижней границей
		if (rank != sizeOfTheGroup - 1)
		{
		    MPI_Sendrecv(
                dev_Anew + (area_for_one_process - 2) * size + 1, 
				size - 2, 
                MPI_DOUBLE, 
                rank + 1,
                0,
			    dev_Anew + (area_for_one_process - 1) * size + 1, 
				size - 2, 
                MPI_DOUBLE, 
                rank + 1, 
                0, 
                MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE
            );
		}

        ++num_of_iterations;
        std::swap(dev_A, dev_Anew);
    }

    printf("Iterations: %d\nAccuracy: %lf\n", num_of_iterations, accuracy);

    // free memory section
    // GPU free
    cudaFree(dev_A);
    CUDA_CHECK("free dev_A");
    cudaFree(dev_Anew);
    CUDA_CHECK("free dev_Anew");
    cudaFree(buff);
    CUDA_CHECK("free buff");
    cudaFree(d_temp_storage);
    CUDA_CHECK("free d_temp_storage");

    // CPU free
    free(A);
    free(Anew);

    MPI_Finalize();
    MPI_CHECK("mpi finalize");

    return 0;
}
