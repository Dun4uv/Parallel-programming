#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <mpi.h>

using namespace std;

using Matrix = vector<double>;

Matrix readMatrix(const string& filename, int& N) {
    ifstream file(filename);
    if (!file) {
        cerr << "Ошибка при открытии файла: " << filename << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            cout << "Использование:\n";
            cout << "mpirun -np <cores> ./matrix_mul_mpi A.txt B.txt result.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = 0;
    Matrix A, B, C;
    double start_time, end_time;

    if (rank == 0) {
        int N2;
        A = readMatrix(argv[1], N);
        B = readMatrix(argv[2], N2);

        if (N != N2) {
            cerr << "Размеры матриц не совпадают!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        C.resize(N * N, 0.0);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B.resize(N * N);
    }

    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<int> sendcounts(size, 0);
    vector<int> displs(size, 0);

    int rows_per_proc = N / size;
    int remainder = N % size;
    int sum = 0;

    for (int i = 0; i < size; i++) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    int local_elements = sendcounts[rank];
    int local_rows = local_elements / N;
    Matrix local_A(local_elements);
    Matrix local_C(local_elements, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_A.data(), local_elements, MPI_DOUBLE,
        0, MPI_COMM_WORLD);


    for (int i = 0; i < local_rows; i++) {
        for (int k = 0; k < N; k++) {
            double a_ik = local_A[i * N + k];
            for (int j = 0; j < N; j++) {
                local_C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }

    MPI_Gatherv(local_C.data(), local_elements, MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        writeMatrix(argv[3], C, N);

        long long operations = 2LL * N * N * N;
        cout << "Размер матрицы: " << N << "x" << N << '\n';
        cout << "Операции: " << operations << '\n';
        cout << "Время: " << (end_time - start_time) << " seconds\n";
        cout << "Процессов MPI: " << size << '\n';
    }

    MPI_Finalize();
    return 0;
}