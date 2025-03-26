#include "../common/book.h"
#define N 10000

void add(int *a, int *b, int *c, int size) {
    int tid = 0;
    while (tid < size) {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

int main(void) {
    int a_1[N/2], b_1[N/2], c_1[N/2];
    int a_2[N/2], b_2[N/2], c_2[N/2];

    for (int i = 0; i < N/2; i++) {
        a_1[i] = -i;
        b_1[i] = i * i;
        a_2[i] = -(i + N/2);
        b_2[i] = (i + N/2) * (i + N/2);
    }

    clock_t start = clock();

    add(a_1, b_1, c_1, N/2);
    add(a_2, b_2, c_2, N/2);

    clock_t end = clock();

    printf("소요 시간: %lf 초\n", (double)(end - start) / CLOCKS_PER_SEC);

    return 0;
}
