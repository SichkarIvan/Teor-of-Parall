#include <stdio.h>
#include <cmath>
#include <chrono>

#define N 10000000
#ifdef double_
using type = double;
#else
using type = float;
#endif

type arr[N];

int main(){
	auto allProgramStart = std::chrono::high_resolution_clock::now();
	type pi = acos(-1);
	type temp = 2 * pi / N;
	type res = 0;
	
	#pragma acc data create(arr[:N]) copy(res) copyin(temp)
	{
		auto C1 = std::chrono::high_resolution_clock::now();
		#pragma acc kernels
		for (int i = 0; i < N; i++)
			arr[i] = sin(temp * i);
		long long R1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-C1).count();
		printf("Fst = %lli\n", R1);
		#pragma acc kernels
		auto C2 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < N; i++)
			res += arr[i];
		long long R2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-C2).count();
		printf("Snd = %lli\n", R2);
		
	}
	
	auto allProgramElapsed = std::chrono::high_resolution_clock::now() - allProgramStart;
	long long programMicro = std::chrono::duration_cast<std::chrono::microseconds>(allProgramElapsed).count();

	printf("\n%0.20lf\n%lli\n", res, programMicro);
	return 0;
}


