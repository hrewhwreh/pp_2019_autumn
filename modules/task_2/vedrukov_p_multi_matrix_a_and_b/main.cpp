// Copyright 2019 Vedrukov Pavel
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <vector>
#include "./multiplicate_matrix.h"

TEST(test_1, 4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A;
    std::vector<int> B;
    A = get_random_matrix(4);
    B = get_random_matrix(4);
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 4, 4, 4);
    std::vector<int> result;
    result = simple_alg(A, B, 4, 4, 4, 4);
    if (rank == 0) {
        ASSERT_EQ(C, result);
    }
}

TEST(test_2, 8) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A;
    std::vector<int> B;
    A = get_random_matrix(8);
    B = get_random_matrix(8);
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 8, 8, 8);
    std::vector<int> result;
    result = simple_alg(A, B, 8, 8, 8, 8);
    if (rank == 0) {
        ASSERT_EQ(C, result);
    }
}

TEST(test_3, 20) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A;
    std::vector<int> B;
    A = get_random_matrix(20);
    B = get_random_matrix(20);
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 20, 20, 20);
    std::vector<int> result;
    result = simple_alg(A, B, 20, 20, 20, 20);
    if (rank == 0) {
        ASSERT_EQ(C, result);
    }
}

TEST(test_4, 40) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A;
    std::vector<int> B;
    A = get_random_matrix(40);
    B = get_random_matrix(40);
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 40, 40, 40);
    std::vector<int> result;
    result = simple_alg(A, B, 40, 40, 40, 40);
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
