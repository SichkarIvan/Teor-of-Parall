#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <chrono>

using namespace std;

double eps = 1E-6;
int iter_max = 1E6;
int size = 128;

int main(int argc, char *argv[]){
    auto start_time = chrono::high_resolution_clock::now();
    for(int arg = 0; arg < argc; arg++){ // Ввод данных
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
 
    double** F = new double*[size];
    for(size_t i = 0; i < size; ++i) 
        F[i] = new double[size];
    double** Fnew = new double*[size];
    for(size_t i = 0; i < size; ++i) 
        Fnew[i] = new double[size];

    for(int i = 0; i < size; i++) {
                F[0][i] = 10 / size * i + 10;
                F[i][0] = 10 / size * i + 10;
                F[size-1][i] = 10 / size * i + 20;
                F[i][size-1] = 10 / size * i + 20;
            }

    double error = 1;
    int iteration = 0;

#ifdef NVPROF_
    nvtxRangePush("MainCycle");
#endif

#pragma acc data copy(F[:size][:size]) create(Fnew[:size][:size])
    while (error > eps && iteration < iter_max )
    {
        error = 0.0;

#pragma acc parallel loop reduction(max:error)
        for( int j = 1; j < size-1; j++) {
#pragma acc loop reduction(max:error)
            for( int i = 1; i < size-1; i++ ) {
                Fnew[j][i] = 0.25 * ( F[j][i+1] + F[j][i-1] + F[j-1][i] + F[j+1][i]);
                error = fmax( error, fabs(Fnew[j][i] - F[j][i]));
            }
        }
#pragma acc parallel loop
        for( int j = 1; j < size-1; j++) {
#pragma acc loop 
            for( int i = 1; i < size-1; i++ ) {
                F[j][i] = Fnew[j][i];
            }
        }
        
        iteration++;
    }

#ifdef NVPROF_
    nvtxRangePop();
#endif

    cout << "Iterations: " << iteration << endl;
    cout << "Error: " << error << endl;

    for(size_t i = 0; i < size; ++i) 
        delete[] F[i];
    delete[] F;

    for(size_t i = 0; i < size; ++i) 
        delete[] Fnew[i];
    delete[] Fnew;

    cout << "Total time = " << chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() << endl;

    return 0;
}