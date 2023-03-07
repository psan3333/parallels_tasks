#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <memory.h>

int main(int argc, char* argv[]) {

    //размер сетки
    int size = atoi(argv[2]);

    //проверка правильности ввода значений
    //количество итераций
    int max_iter_input = atoi(argv[3]);
    int iter = 0, max_iter = 1000000;
    if (max_iter_input > max_iter){ //проверка, не превышает ли ввод предельное число итераций max_iter
        printf("Count of iterations mustn't exceed 10^6 operations");
        return 0;
    }

    //точность
    double precision = atof(argv[1]);
    double max_precision = 0.000001, error = precision + 1.0;
    if (precision < max_precision) { //проверка, не превышает ли ввод предельную точность max_precision
        printf("Precision mustn't be lower than 10^-6");
        return 0;
    } 

    //сетка хранится в одномерном массиве размера size*size 
    double* A = (double*)calloc(size * size, sizeof(double));
    double* Anew = (double*)calloc(size * size, sizeof(double));

    memset(A, 0, size * size * sizeof(double));
    memset(Anew, 0, size * size * sizeof(double));

	
    //инициализация сетки (значения в углах: 10, 20, 30, 20 - с левого нижнего края против часовой стрелки)
    A[0] = 10.0;
    A[size - 1] = 20.0;
    A[size * size - 1] = 30.0;
    A[size * (size - 1)] = 20.0;

    Anew[0] = 10.0;
    Anew[size - 1] = 20.0;
    Anew[size * size - 1] = 30.0;
    Anew[size * (size - 1)] = 20.0;

    #pragma acc data copy(A[0:size*size], Anew[0:size*size]) //копирование данных с CPU на GPU
    {
        clock_t start = clock(); //побочный таймер
        double step = 10.0 / (size - 1); //шаг инициализации краёв сетки

        #pragma acc parallel loop vector worker num_workers(4) vector_length(32)
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
        
        while (error > precision && iter < max_iter_input) {

            error = 0.0;
            ++iter;

            #pragma acc data present(A, Anew)
            #pragma acc parallel loop vector vector_length(256) gang num_gangs(256) collapse(2) reduction(max:error) async(1)
            for (int i = 1; i < size - 1; i++){
                for (int j = 1; j < size - 1; j++) {
                    Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);
                    error = fmax(error, fabs(Anew[i * size + j] - A[i * size + j]));
                }
            }
            #pragma acc wait

            double* temp = A;
            A = Anew;
            Anew = temp;

        }
        
        clock_t end = clock(); //конец отсчёта таймера
        printf("%lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    }

    printf("%d %lf\n", iter, error); //вывод: количество итераций 'iter' основного цикла 'while' и полученная ошибка 'error'

    //освобождение памяти
    free(Anew);
    free(A);

    return 0;
}
