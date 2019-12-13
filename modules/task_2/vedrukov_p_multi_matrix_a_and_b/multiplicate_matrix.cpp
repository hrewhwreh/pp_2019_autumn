// Copyright 2019 Vedrukov Pavel

#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <iostream>
#include "../../../modules/task_2/vedrukov_p_multi_matrix_a_and_b/multiplicate_matrix.h"

std::vector<int> multiplicate_matrix(std::vector<int> A, std::vector<int> B,
                                     int r_size_A, int c_size_A,
                                     int r_size_B, int c_size_B) {
    int Proc_num;
    int Proc_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &Proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &Proc_rank);
    MPI_Status status;

    if (r_size_A != c_size_B) {
        throw "size error";
    }

    if (Proc_num <= 2) {
        std::vector<int> C;
        C.resize(c_size_A * r_size_B);
        for (int i = 0; i < static_cast<int>(C.size()); i++) {
            for (int j = 0; j < static_cast<int>(C.size()); j++) {
                for (int k = 0; k < static_cast<int>(C.size()); k++) {
                    C[i * c_size_A + j] += A[i * r_size_A + k] * B[k * c_size_B + j];
                }
            }
        }
        return C;
    } else {
        int c_size = c_size_A;
        int r_size = r_size_B;
        if (c_size_A % Proc_num != 0) {
            c_size = c_size_A + Proc_num - c_size_A % Proc_num;
        }
        if (r_size_B % Proc_num != 0) {
            r_size = r_size_B + Proc_num - r_size_B % Proc_num;
        }

        std::vector<int> C;
        C.resize(c_size * r_size);
        if (Proc_rank == 0) {
            A.resize(c_size * r_size_A);
            std::vector<double> temp;
            temp.resize(r_size_B * c_size_B);
            for (int i = 0; i < c_size_B; i++) {
                for (int j = 0; j < r_size_B; j++) {
                    temp[j * c_size_B + i] = B[i * r_size_B + j];
                }
            }
            for (int i = 0; i < c_size_B; i++) {
                for (int j = 0; j < r_size_B; j++) {
                    B[j * c_size_B + i] = temp[j * c_size_B + i];
                }
            }
            B.resize(r_size * c_size_B);
        }

        std::vector<int> buf_A;
        std::vector<int> buf_B;
        std::vector<int> buf_C;
        int buf_A_c_size = c_size / Proc_num;
        int buf_B_r_size = r_size / Proc_num;
        buf_A.resize(buf_A_c_size * r_size_A);
        buf_B.resize(c_size_B * buf_B_r_size);
        buf_C.resize(c_size * buf_B_r_size);
        int part_A = buf_A_c_size * r_size_A;
        int part_B = buf_B_r_size * c_size_B;

        MPI_Scatter(&A, part_A, MPI_INT, &buf_A, part_A, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(&B, part_B, MPI_INT, &buf_B, part_B, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < part_A; i++) {
            for (int j = 0; j < part_B; j++) {
                for (int k = 0; k < r_size_A; k++) {
                    buf_C[i * r_size + j + Proc_rank * buf_B_r_size] = buf_A[i * r_size + k] * buf_B[j * c_size + k];
                }
            }
        }

        int n_p, p_p;
        for (int a = 0; a < Proc_num; a++) {
            n_p = Proc_rank + 1;
            if (Proc_rank == Proc_num - 1) {
                n_p = 0;
            }
            p_p = Proc_rank - 1;
            if (Proc_rank == 0) {
                p_p = Proc_num - 1;
            }
            MPI_Sendrecv_replace(&buf_B, part_B, MPI_INT, n_p, 0, p_p, 0, MPI_COMM_WORLD, &status);
            int t = 0;
            for (int i = 0; i < part_A; i++) {
                for (int j = 0; j < part_B; j++) {
                    for (int k = 0; k < r_size_A; k++) {
                        t = buf_A[i * r_size + k] * buf_B[j * c_size + k];
                    }
                    if (Proc_rank - a >= 0) {
                        buf_C[i * r_size + j + (Proc_rank - a) * buf_B_r_size] = t;
                    } else {
                        buf_C[i * r_size + j + (Proc_num + Proc_rank - a) * buf_B_r_size] = t;
                    }
                }
            }
        }
        MPI_Gather(&buf_C, r_size * buf_B_r_size, MPI_INT, &C, r_size * buf_B_r_size, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> result;
        result.resize(c_size_A * r_size_B);
        for (int i = 0; i < c_size_A; i++) {
            for (int j = 0; j < r_size_B; j++) {
                result[i * r_size_B + j] = C[i * r_size_B + j];
            }
        }
        return result;
    }
}

std::vector<int> get_random_matrix(int size) {
    std::vector<int> A;
    A.resize(size * size);
    for (int i = 0; i < size * size; i++) {
        A[i] = -10 + rand_r(static_cast<unsigned int>size) % 21;
    }
    return A;
}

std::vector<int> simple_alg(std::vector<int> A, std::vector<int> B,
                                     int r_size_A, int c_size_A,
                                     int r_size_B, int c_size_B) {
    std::vector<int> C;
    C.resize(c_size_A * r_size_B);
    for (int i = 0; i < static_cast<int>(C.size()); i++) {
        for (int j = 0; j < static_cast<int>(C.size()); j++) {
            for (int k = 0; k < static_cast<int>(C.size()); k++) {
                C[i * r_size_B + j] += A[i * r_size_A + k] * B[k * c_size_B + j];
            }
        }
    }
    return C;
}
