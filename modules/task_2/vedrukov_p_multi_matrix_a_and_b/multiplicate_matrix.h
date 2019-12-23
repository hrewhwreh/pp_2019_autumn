// Copyright 2019 Vedrukov Pavel
#ifndef MODULES_TASK_2_VEDRUKOV_P_MULTI_MATRIX_A_AND_B_MULTIPLICATE_MATRIX_H_
#define MODULES_TASK_2_VEDRUKOV_P_MULTI_MATRIX_A_AND_B_MULTIPLICATE_MATRIX_H_

#include <iostream>
#include <vector>

std::vector<int> multiplicate_matrix(std::vector<int> A, std::vector<int> B,
                                     int size, int c_size_A,
                                     int r_size_B);
std::vector<int> get_random_matrix(int size);
std::vector<int> simple_alg(std::vector<int> A, std::vector<int> B,
                                     int r_size_A, int c_size_A,
                                     int r_size_B, int c_size_B);

#endif  // MODULES_TASK_2_VEDRUKOV_P_MULTI_MATRIX_A_AND_B_MULTIPLICATE_MATRIX_H_
