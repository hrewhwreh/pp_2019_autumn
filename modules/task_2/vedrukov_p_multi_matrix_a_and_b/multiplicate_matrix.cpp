// Copyright 2019 Vedrukov Pavel

#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include "../../../modules/task_2/vedrukov_p_multi_matrix_a_and_b/multiplicate_matrix.h"

std::vector<int> multiplicate_matrix(std::vector<int> A, std::vector<int> B,
                                     int size, int c_size_A,
                                     int r_size_B) {
    int Proc_num;
    int Proc_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &Proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &Proc_rank);
    MPI_Status status;
    if (Proc_num < 2) {
        std::vector<int> C;
        C.resize(c_size_A * r_size_B);
        for (int i = 0; i < c_size_A; i++) {
            for (int j = 0; j < r_size_B; j++) {
                for (int k = 0; k < size; k++) {
                    C[i * c_size_A + j] += A[i * size + k] * B[k * r_size_B + j];
                }
            }
        }
        return C;
    } else {
        int c_size = c_size_A;
        int r_size = r_size_B;
        if (c_size_A % Proc_num != 0) {
            c_size = c_size_A + Proc_num - (c_size_A % Proc_num);
        }
        if (r_size_B % Proc_num != 0) {
            r_size = r_size_B + Proc_num - (r_size_B % Proc_num);
        }

        std::vector<int> _A;
        std::vector<int> _B;
        std::vector<int> C;
        C.resize(c_size * r_size);
        if (Proc_rank == 0) {
            _A.resize(c_size * size);
            _B.resize(size * r_size_B);
            std::vector<int> temp;
            temp.resize(r_size_B * size);
            for (int i = 0; i < c_size_A; i++) {
                for (int j = 0; j < size; j++) {
                    _A[i * size + j] = A[i * size + j];
                }
            }
            for (int i = c_size_A; i < c_size; i++) {
                for (int j = 0; j < size; j++) {
                    A[i * size + j] = 0;
                }
            }
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < r_size_B; j++) {
                    temp[j * size + i] = B[i * r_size_B + j];
                }
            }
            for (int i = 0; i < r_size_B * size; i++) {
                _B[i] = temp[i];
            }
            _B.resize(size * r_size);
            for (int i = r_size_B; i < r_size; i++) {
                for (int j = 0; j < size; j++) {
                    _B[i * size + j] = 0;
                }
            }
        }

        std::vector<int> buf_A;
        std::vector<int> buf_B;
        std::vector<int> buf_C;
        int buf_A_c_size = c_size / Proc_num;
        int buf_B_r_size = r_size / Proc_num;
        buf_A.resize(buf_A_c_size * size);
        buf_B.resize(size * buf_B_r_size);
        buf_C.resize(r_size * buf_A_c_size);
        int part_A = buf_A_c_size * size;
        int part_B = buf_B_r_size * size;
        if (Proc_rank == 0) {
            MPI_Scatter(&_A[0], part_A, MPI_INT, &buf_A[0], part_A, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatter(&_B[0], part_B, MPI_INT, &buf_B[0], part_B, MPI_INT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Scatter(NULL, 0, MPI_INT, &buf_A[0], part_A, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatter(NULL, 0, MPI_INT, &buf_B[0], part_B, MPI_INT, 0, MPI_COMM_WORLD);
        }
        int tmp = 0;
        for (int i = 0; i < buf_A_c_size; i++) {
            for (int j = 0; j < buf_B_r_size; j++) {
                for (int k = 0; k < size; k++) {
                    tmp += buf_A[i * size + k] * buf_B[j * size + k];
                }
                buf_C[i * r_size + j + Proc_rank * buf_B_r_size] = tmp;
                tmp = 0;
            }
        }

        int n_p, p_p;
        for (int a = 1; a < Proc_num; a++) {
            n_p = Proc_rank + 1;
            if (Proc_rank == Proc_num - 1) {
                n_p = 0;
            }
            p_p = Proc_rank - 1;
            if (Proc_rank == 0) {
                p_p = Proc_num - 1;
            }
            MPI_Sendrecv_replace(&buf_B[0], part_B, MPI_INT, n_p, 0, p_p, 0, MPI_COMM_WORLD, &status);
            int t = 0;
            for (int i = 0; i < buf_A_c_size; i++) {
                for (int j = 0; j < buf_B_r_size; j++) {
                    for (int k = 0; k < size; k++) {
                        t += buf_A[i * size + k] * buf_B[j * size + k];
                    }
                    if (Proc_rank - a >= 0) {
                        buf_C[i * r_size + j + (Proc_rank - a) * buf_B_r_size] = t;
                    } else {
                        buf_C[i * r_size + j + (Proc_num + Proc_rank - a) * buf_B_r_size] = t;
                    }
                    t = 0;
                }
            }
        }
        MPI_Gather(&buf_C[0], r_size * buf_A_c_size, MPI_INT, &C[0], r_size * buf_A_c_size, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<int> result;
        result.resize(c_size_A * r_size_B);
        for (int i = 0; i < c_size_A; i++) {
            for (int j = 0; j < r_size_B; j++) {
                result[i * r_size_B + j] = C[i * r_size + j];
            }
        }
        return result;
    }
}

std::vector<int> get_random_matrix(int size) {
    std::vector<int> A;
    A.resize(size * size);
    std::mt19937 gen(time(0));
    std::uniform_int_distribution<int> uid1(-10, 10);
    for (int i = 0; i < size * size; i++) {
        A[i] = uid1(gen);
    }
    return A;
}

std::vector<int> simple_alg(std::vector<int> A, std::vector<int> B,
                                     int r_size_A, int c_size_A,
                                     int r_size_B, int c_size_B) {
    std::vector<int> C;
    C.resize(c_size_A * r_size_B);
    for (int i = 0; i < c_size_A; i++) {
        for (int j = 0; j < r_size_B; j++) {
            for (int k = 0; k < r_size_A; k++) {
                C[i * r_size_B + j] += A[i * r_size_A + k] * B[k * c_size_B + j];
            }
        }
    }
    return C;
}
