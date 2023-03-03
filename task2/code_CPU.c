#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char* argv[]) {

    int size = atoi(argv[2]);

    double** A = (double**)malloc(size * sizeof(double*));
    double** Anew = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        A[i] = (double*)calloc(size, sizeof(double));
        Anew[i] = (double*)calloc(size, sizeof(double));
    }

    int max_iter_input = atoi(argv[3]);
    int iter = 0, max_iter = 1000000;
    if (max_iter_input > max_iter){
        printf("Count of iterations mustn't exceed 10^6 operations");
        return 0;
    }

    double precision = atof(argv[1]);
    double error = 1.0, max_precision = 0.000001;
    if (precision < max_precision) {
        printf("Precision mustn't be lower than 10^-6");
        return 0;
    } 

    A[size - 1][0] = 10.0;
    A[size - 1][size - 1] = 20.0;
    A[0][size - 1] = 30.0;
    A[0][0] = 20.0;
    Anew[size - 1][0] = 10.0;
    Anew[size - 1][size - 1] = 20.0;
    Anew[0][size - 1] = 30.0;
    Anew[0][0] = 20.0;


    for (int i = 1; i < size - 1; i++) {
        A[size - 1][i] = A[size - 1][0] + (double)10.0 * i / (size - 1);
        Anew[size - 1][i] = A[size - 1][i];

        A[0][i] = A[0][0] + (double)10.0 * i / (size - 1);
        Anew[0][i] = A[0][i];

        A[size - 1 - i][size - 1] = A[size  - 1][size - 1] + (double)10.0 * i / (size - 1);
        Anew[size - 1 - i][size - 1] = A[size - 1 - i][size - 1];

        A[size - 1 - i][0] = A[size - 1][0] + (double)10.0 * i / (size - 1);
        Anew[size - 1 - i][0] = A[size - 1 - i][0];
    }

    while (error > precision && iter < max_iter_input) {

        error = 0.0;

        ++iter;

        for (int j = 1; j < size - 1; j++){
            for (int i = 1; i < size - 1; i++) {
                if (iter % 2) {
                    Anew[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]);
                }
                else {
                    A[i][j] = 0.25 * (Anew[i + 1][j] + Anew[i - 1][j] + Anew[i][j - 1] + Anew[i][j + 1]);
                }
                error = fmax(error, fabs(Anew[j][i] - A[j][i]));
            }
        }
    }

    printf("%d %lf\n", iter, error);
    
    for (int i = 0; i < size; i++) {
        free(A[i]);
        free(Anew[i]);
    }

    free(Anew);
    free(A);

    return 0;
}
