#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

using namespace std;

using Matrix = vector<double>;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "Ошибка GPU: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

Matrix readMatrix(const string& filename, int& N) {
    ifstream file(filename);
    if (!file) {
        cerr << "Ошибка при открытии файла: " << filename << '\n';
        exit(1);
    }

    file >> N;
    Matrix M(N * N);

    for (int i = 0; i < N * N; i++)
        file >> M[i];

    return M;
}

void writeMatrix(const string& filename, const Matrix& M, int N) {
    ofstream file(filename);
    file << N << "\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            file << M[i * N + j] << " ";
        file << "\n";
    }
}

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cout << "Использование:\n";
        cout << "./matrix_mul_cuda A.txt B.txt result.txt <block_size>\n";
        return 1;
    }

    int N = 0, N2 = 0;
    Matrix A = readMatrix(argv[1], N);
    Matrix B = readMatrix(argv[2], N2);
    int block_size = stoi(argv[4]);

    if (N != N2) {
        cerr << "Размеры матриц не совпадают!\n";
        return 1;
    }

    Matrix C(N * N, 0.0);
    size_t bytes = N * N * sizeof(double);

    double* d_A, * d_B, * d_C;
    cudaCheckError(cudaMalloc(&d_A, bytes));
    cudaCheckError(cudaMalloc(&d_B, bytes));
    cudaCheckError(cudaMalloc(&d_C, bytes));

    cudaCheckError(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    cudaCheckError(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    writeMatrix(argv[3], C, N);

    long long operations = 2LL * N * N * N;
    cout << "Размер матрицы: " << N << "x" << N << '\n';
    cout << "Размер блока: " << block_size << "x" << block_size << '\n';
    cout << "Операции: " << operations << '\n';
    cout << "Время: " << seconds << " seconds\n";

    return 0;
}