#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define at(arr, x, y) (arr[(x)*size+(y)]) 

using namespace std;

// Значения по умолчанию
double eps = 1E-6;
int iter_max = 1E6;
int size = 128;



int main(int argc, char *argv[])
{

// Ввод данных с консоли
    for(int arg = 0; arg < argc; arg++){ 
        stringstream stream;
        if(strcmp(argv[arg], "-error") == 0){
            stream << argv[arg+1];
            stream >> eps;
        }
        else if(strcmp(argv[arg], "-iter") == 0){
            stream << argv[arg+1];
            stream >> iter_max;
        }
        else if(strcmp(argv[arg], "-size") == 0){
            stream << argv[arg+1];
            stream >> size;
        }
    }
    cout << "Settings:\n\tEPS: " << eps << "\n\tMax iteration: " << iter_max << "\n\tSize: " << size << 'x' << size << "\n\n";
 
    double* F = new double[size*size];
    double* Fnew = new double[size*size];


// Ввод данных в грани матрицы F
    for(int i = 0; i < size; i++) {
                at(F, 0, i) = 10 / size * i + 10;
                at(F, i, 0) = 10 / size * i + 10;
                at(F, size-1, i) = 10 / size * i + 20;
                at(F ,i, size-1) = 10 / size * i + 20;
            }


    double error = 1.0;
    int iteration = 0;

    cublasHandle_t handler;
    cublasCreate(&handler);


// Копирование данных F с CPU на GPU, выделение памяти под Fnew
#pragma acc data copyin(F[:size*size]) create(Fnew[:size*size])
{
// Ввод данных в грани матрицы Fnew
#pragma acc parallel loop 
    for( int j = 0; j < size; j++) {
        at(Fnew, j, 0) = at(F, j, 0);
        at(Fnew, 0, j) = at(F, 0, j);
        at(Fnew, j, size-1) = at(F, j, size-1);
        at(Fnew, size-1, j) = at(F, size-1, j);
    }


// Основной цикл
    while (error > eps && iteration < iter_max )
    {

    
// Изменение матрицы Fnew
#pragma acc parallel loop async(0)
        for( int j = 1; j < size-1; j++) {
#pragma acc loop
            for( int i = 1; i < size-1; i++ ) {
                at(Fnew, j, i) = 0.25 * ( at(F, j, i+1) + at(F, j, i-1) + at(F, j-1, i) + at(F, j+1, i));
            }

        }

#pragma acc wait(0)



// Изменение матрицы F
#pragma acc parallel loop async(1)
        for( int j = 1; j < size-1; j++) {
#pragma acc loop
            for( int i = 1; i < size-1; i++ ) {
                at(F, j, i) = 0.25 * ( at(Fnew, j, i+1) + at(Fnew, j, i-1) + at(Fnew, j-1, i) + at(Fnew, j+1, i));
            }

        }

#pragma acc wait(1)

        iteration+=2;


// Каждые 250 итераций проверяем ошибку
        if (iteration % 250 == 0){
            int idx = 0;
            double alpha = -1.0;

// Ищем максимум из разницы
#pragma acc host_data use_device(Fnew, F)
            {
                cublasDaxpy(handler, size * size, &alpha, Fnew, 1, F, 1);
                cublasIdamax(handler, size * size, F, 1, &idx);
            }

// Возвращаем ошибку на host
#pragma acc update host(F[idx - 1]) 
			error = abs(F[idx - 1]);

#pragma acc host_data use_device(Fnew, F)
            cublasDcopy(handler, size * size, Fnew, 1, F, 1);
        }
    }

}


// Выведение результатов и очистка памяти
    cublasDestroy(handler);

    cout << "Iterations: " << iteration << endl;
    cout << "Error: " << error << endl;

    delete[] F;
    delete[] Fnew;

    return 0;

}