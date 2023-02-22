#include <iostream>
#include <cmath>
#include <chrono>

#define N 10000000

#ifdef D
    using type = double;
#else
    using type = float;
#endif

using namespace std;

type arr[N];

int main(){
	auto start_time = chrono::high_resolution_clock::now();

	type temp = 2 * acos(-1) / N;
	type res = 0;
	
	#pragma acc data create(arr[:N]) copy(res) copyin(temp)
	{
		auto start_cycle = chrono::high_resolution_clock::now();
		#pragma acc kernels
		for (int i = 0; i < N; i++) arr[i] = sin(temp * i);

		cout << "First = " << chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-start_cycle).count() << endl;

        start_cycle = chrono::high_resolution_clock::now();
		#pragma acc kernels
		for (int i = 0; i < N; i++) res += arr[i];
		
		cout << "Second = " << chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-start_cycle).count() << endl;
	}

	cout << "Result = " << res << endl;
    cout << "Total time = " << chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start_time).count() << endl;

	return 0;
}


