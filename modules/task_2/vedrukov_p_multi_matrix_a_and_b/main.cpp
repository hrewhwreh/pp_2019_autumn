// Copyright 2019 Vedrukov Pavel
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "./multiplicate_matrix.h"

TEST(freq_symb, zero_freq) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> B{9, 8, 7, 6, 5, 4, 3 ,2 ,1};
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 3, 3, 3, 3);
    std::vector<int> result{30, 24, 18, 84, 69, 54, 138, 114, 90};
    if (rank == 0) {
        ASSERT_EQ(C, result);
    }
}

TEST(freq_symb, full_freq) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A;
    A = get_random_matrix(10);
    std::vector<int> B;
    B = get_random_matrix(10);
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 10, 10, 10, 10);
    std::vector<int> result;
    result = simple_alg(A, B, 10, 10, 10, 10);
    if (rank == 0) {
        ASSERT_EQ(C, result);
    }
}

TEST(count_frequency, some_freq_1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A;
    A = get_random_matrix(100);
    std::vector<int> B;
    B = get_random_matrix(100);
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 100, 100, 100, 100);
    std::vector<int> result;
    result = simple_alg(A, B, 100, 100, 100, 100);
    if (rank == 0) {
        ASSERT_EQ(C, result);
    }
}

TEST(count_frequency, some_freq_2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A;
    A = get_random_matrix(1000);
    std::vector<int> B;
    B = get_random_matrix(1000);
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 1000, 1000, 1000, 1000);
    std::vector<int> result;
    result = simple_alg(A, B, 1000, 1000, 1000, 1000);
    if (rank == 0) {
        ASSERT_EQ(C, result);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
