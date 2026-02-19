#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;
using Matrix = vector<vector<double>>;

Matrix readMatrix(const string& filename, int& N) {
    ifstream file(filename);
    if (!file) {
        cerr << "Ошибка при открытии файла: " << filename << '\n';
        exit(1);
    }

    file >> N;
    Matrix M(N, vector<double>(N));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            file >> M[i][j];

    return M;
}

void writeMatrix(const string& filename, const Matrix& M) {
    ofstream file(filename);
    int N = M.size();

    file << N << "\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            file << M[i][j] << " ";
        file << "\n";
    }
}

Matrix multiply(const Matrix& A, const Matrix& B, int threads) {
    int N = A.size();
    Matrix C(N, vector<double>(N, 0.0));

    omp_set_num_threads(threads);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int main(int argc, char* argv[]) {

    if (argc != 5) {
        cout << "Использование:\n";
        cout << "./matrix_mul A.txt B.txt result.txt threads\n";
        return 1;
    }

    int N1, N2;

    Matrix A = readMatrix(argv[1], N1);
    Matrix B = readMatrix(argv[2], N2);

    if (N1 != N2) {
        cerr << "Размеры матриц не совпадают!\n";
        return 1;
    }

    int threads = stoi(argv[4]);

    auto start = chrono::high_resolution_clock::now();
    Matrix C = multiply(A, B, threads);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    writeMatrix(argv[3], C);

    long long operations = 2LL * N1 * N1 * N1;

    cout << "Размер матрицы: " << N1 << "x" << N1 << '\n';
    cout << "Операции: " << operations << '\n';
    cout << "Время: " << elapsed.count() << " seconds\n";
    cout << "Потоков: " << threads << '\n';

    return 0;
}
