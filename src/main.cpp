#include <iostream>
#include <stdio.h>
#include <omp.h>

int main()
{
	int a[10];

    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        a[i] = 2 * i;
		int tid = omp_get_thread_num();
		printf("assigning i=%d\n", i);
		printf("ThreadId=%i\n", tid);
    }

    return 0;
}
