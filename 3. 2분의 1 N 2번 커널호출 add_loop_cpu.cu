#include "../common/book.h"
#define N 10000

void add(int *a, int *b, int *c, int start, int end) {
    int tid = start;
    while (tid < end) {
        c[tid] = a[tid] + b[tid];
        tid += 1;
    }
}

int main(void) {
    int a[N], b[N], c[N];
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    clock_t start = clock();
    add(a, b, c, 0, N / 2);     // 첫 번째 절반 연산
    add(a, b, c, N / 2, N);     // 두 번째 절반 연산
    clock_t end = clock();

    printf("소요시간: %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    // 결과 확인
    for (int i = 0; i < 10; i++) {  // 처음 10개만 출력
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}

