cat > cuda_test.cu << EOF
#include <stdio.h>

__global__ void hello_cuda() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU\n");
    hello_cuda<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF
