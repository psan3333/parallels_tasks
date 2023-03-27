#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <assert.h>
#include <cublas_v2.h>


int main(int argc, char* argv[]) {

    //parse command prompt arguments
    int size = std::stoi(argv[2]);
    int max_iter_input = std::stoi(argv[3]);
    int max_iter = 1000000;
    if (max_iter_input > max_iter) {
        std::cerr << "Count of iterations mustn't exceed 10^6 operations" << std::endl;
        exit(EXIT_FAILURE);
    }

    double precision = std::stod(argv[1]);
    double max_precision = 0.000001;
    if (precision < max_precision) {
        std::cerr << "Precision mustn't be lower than 10^-6" << std::endl;
        exit(EXIT_FAILURE);
    }

    //allocate memory and set A = 0
    size_t matrixSize = size * size;
    double* A = new double[matrixSize];
    double* Anew = new double[matrixSize];
    std::memset(A, 0, matrixSize * sizeof(double));

    //init angles values
    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[size * size - 1] = 30.0;
    A[size * (size - 1)] = 20.0;

    Anew[0] = 10.0;
    Anew[size - 1] = 20.0;
    Anew[size * size - 1] = 30.0;
    Anew[size * (size - 1)] = 20.0;

    //init data for main loop execution
    double step = 10.0 / (size - 1);
    int iter = 0;
    double error = precision + 1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);

    #pragma acc enter data copyin(A[:matrixSize], Anew[:matrixSize])
    {

        //interpolation for angles
        #pragma acc parallel loop
        for (int i = 1; i < size - 1; i++)
        {
            A[i] = A[0] + i * step;
            A[i * size] = A[0] + i * step;
            A[size - 1 + size * i] = A[size - 1] + i * step;
            A[size * (size - 1) + i] = A[size * (size - 1)] + i * step;

            Anew[i] = Anew[0] + i * step;
            Anew[i * size] = Anew[0] + i * step;
            Anew[size - 1 + size * i] = Anew[size - 1] + i * step;
            Anew[size * (size - 1) + i] = Anew[size * (size - 1)] + i * step;
        }

        //main loop
        while (iter < max_iter_input && error > precision) {

            //calculating average values
            #pragma acc data present(A[:matrixSize], Anew[:matrixSize]) //updating pointers on GPU
            #pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256)
            for (int i = 1; i < size - 1; i++) {
                for (int j = 1; j < size - 1; j++) {
                    Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);
                }
            }
            
            //updating error only 1/100 of main loop iterations
            if (iter % 100 == 0 || iter + 1 == max_iter_input) {
                error = 0.0;
                double a = -1;
                int idxMax = 0;
                
                cublasStatus_t stat1, stat2, stat3;
                #pragma acc host_data use_device(A, Anew)
                {
                    stat1 = cublasDaxpy(handle, matrixSize, &a, Anew, 1, A, 1);
                    stat2 = cublasIdamax(handle, matrixSize, A, 1, &idxMax);
                }

                //check for failure
                if(stat1 != CUBLAS_STATUS_SUCCESS)
                    exit(EXIT_FAILURE);
                if(stat2 != CUBLAS_STATUS_SUCCESS)
                    exit(EXIT_FAILURE);

                #pragma acc update host(A[idxMax - 1])
                error = std::abs(A[idxMax - 1]);

                #pragma acc host_data use_device(A, Anew)
                {
                    stat3 = cublasDcopy(handle, matrixSize, Anew, 1, A, 1);
                }
                if(stat3 != CUBLAS_STATUS_SUCCESS)
                    exit(EXIT_FAILURE);
            }

            std::swap(A, Anew);
            iter++;
        }

        printf("Iterations: %d\nPrecision: %lf\n", iter, error);
    }

    cublasDestroy(handle);
    #pragma acc exit data delete(A[:matrixSize], Anew[:matrixSize])
    delete[] Anew;
    delete[] A;

    return 0;
}
