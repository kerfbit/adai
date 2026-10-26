// TD-423: checked binary reads and writes for checkpoint sidecars.
#include "../src/BinaryIO.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {
const std::string kPath = "/tmp/adai_binaryio_test.bin";
}

TEST(BinaryIOTest, RoundTripsValuesAndMatrices) {
    Matrix m(2, 3);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            m(r, c) = static_cast<float>(r * 10 + c) + 0.5f;
    {
        adai::binio::Writer w(kPath);
        w.value(42, "int");
        w.value(1.25f, "float");
        w.matrix(m, "matrix");
        w.close();
    }
    adai::binio::Reader r(kPath);
    EXPECT_EQ(r.value<int>("int"), 42);
    EXPECT_FLOAT_EQ(r.value<float>("float"), 1.25f);
    const Matrix back = r.matrix(2, 3, "matrix");
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(back(i, j), m(i, j));
    EXPECT_NO_THROW(r.expect_end());
    std::filesystem::remove(kPath);
}

TEST(BinaryIOTest, ShortReadThrows) {
    {
        adai::binio::Writer w(kPath);
        w.value(7, "int");
        w.close();
    }
    adai::binio::Reader r(kPath);
    EXPECT_EQ(r.value<int>("int"), 7);
    EXPECT_THROW(r.value<int>("missing int"), std::runtime_error);
    adai::binio::Reader again(kPath);
    EXPECT_THROW(again.matrix(1, 4, "matrix"), std::runtime_error);
    std::filesystem::remove(kPath);
}

TEST(BinaryIOTest, TrailingDataThrows) {
    {
        adai::binio::Writer w(kPath);
        w.value(7, "int");
        w.value(8, "int");
        w.close();
    }
    adai::binio::Reader r(kPath);
    r.value<int>("int");
    EXPECT_THROW(r.expect_end(), std::runtime_error);
    std::filesystem::remove(kPath);
}

TEST(BinaryIOTest, MissingFileThrows) {
    EXPECT_THROW(adai::binio::Reader("/tmp/adai_binaryio_does_not_exist.bin"), std::runtime_error);
    EXPECT_THROW(adai::binio::Writer("/nonexistent_dir_adai/x.bin"), std::runtime_error);
}

TEST(BinaryIOTest, FailedWriteThrows) {
    if (!std::filesystem::exists("/dev/full"))
        GTEST_SKIP() << "no /dev/full";
    Matrix big(64, 1024);
    adai::binio::Writer w("/dev/full");
    // The stream buffers, so the failure surfaces at the latest on close().
    EXPECT_THROW(
        {
            w.matrix(big, "matrix");
            w.close();
        },
        std::runtime_error);
}
