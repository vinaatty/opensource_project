#include "../common/book.h"
#define N 10000

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x;
    if (tid < N/2) {
        c[tid] = a[tid] + b[tid];
    }
}

int main(void) {
    int a_1[N/2], b_1[N/2], c_1[N/2];
    int a_2[N/2], b_2[N/2], c_2[N/2];

    int *dev_a_1, *dev_b_1, *dev_c_1;
    int *dev_a_2, *dev_b_2, *dev_c_2;

    HANDLE_ERROR(cudaMalloc((void**)&dev_a_1, N/2 * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b_1, N/2 * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c_1, N/2 * sizeof(int)));

    HANDLE_ERROR(cudaMalloc((void**)&dev_a_2, N/2 * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b_2, N/2 * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c_2, N/2 * sizeof(int)));

    for (int i = 0; i < N/2; i++) {
        a_1[i] = -i;
        b_1[i] = i * i;
        a_2[i] = -(i + N/2);
        b_2[i] = (i + N/2) * (i + N/2);
    }

    HANDLE_ERROR(cudaMemcpy(dev_a_1, a_1, N/2 * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b_1, b_1, N/2 * sizeof(int), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(dev_a_2, a_2, N/2 * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b_2, b_2, N/2 * sizeof(int), cudaMemcpyHostToDevice));

    clock_t start = clock();

    add<<<N/2, 1>>>(dev_a_1, dev_b_1, dev_c_1);
    cudaDeviceSynchronize();

    add<<<N/2, 1>>>(dev_a_2, dev_b_2, dev_c_2);
    cudaDeviceSynchronize();

    clock_t end = clock();

    printf("소요 시간: %lf 초\n", (double)(end - start) / CLOCKS_PER_SEC);

    HANDLE_ERROR(cudaMemcpy(c_1, dev_c_1, N/2 * sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(c_2, dev_c_2, N/2 * sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_a_1));
    HANDLE_ERROR(cudaFree(dev_b_1));
    HANDLE_ERROR(cudaFree(dev_c_1));

    HANDLE_ERROR(cudaFree(dev_a_2));
    HANDLE_ERROR(cudaFree(dev_b_2));
    HANDLE_ERROR(cudaFree(dev_c_2));

    return 0;
}
