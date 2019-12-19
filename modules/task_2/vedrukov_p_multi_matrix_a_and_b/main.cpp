// Copyright 2019 Vedrukov Pavel
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "./multiplicate_matrix.h"

TEST(test_1, any) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> A{1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int> B{9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<int> C;
    C = multiplicate_matrix(A, B, 3, 3, 3, 3);
    std::vector<int> result{18, 15, 12, 18, 15, 12, 18, 15, 12};
    if (rank == 0) {
        ASSERT_EQ(C, result);
    }
}

TEST(test_2, 10) {
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

TEST(test_3, 100) {
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

TEST(test_4, 1000) {
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
