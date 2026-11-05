// Tests for adai::write_file_atomically (TD-495).

#include "AtomicFile.hpp"
#include <gtest/gtest.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {
std::string read_all(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

class AtomicFileTest : public ::testing::Test {
   protected:
    fs::path dir_;
    void SetUp() override {
        dir_ = fs::temp_directory_path() / ("adai_atomicfile_" + std::to_string(::getpid()));
        fs::remove_all(dir_);
        fs::create_directories(dir_);
    }
    void TearDown() override {
        std::error_code ec;
        fs::permissions(dir_, fs::perms::owner_all, fs::perm_options::add, ec);
        fs::remove_all(dir_, ec);
    }
    bool only_file_is(const fs::path& name) {
        int n = 0;
        for (const auto& e : fs::directory_iterator(dir_)) {
            ++n;
            if (e.path().filename() != name)
                return false;
        }
        return n == 1;
    }
};
}  // namespace

TEST_F(AtomicFileTest, WritesAndReplaces) {
    const fs::path p = dir_ / "sub" / "queue.txt";
    EXPECT_EQ(adai::write_file_atomically(p.string(), "first\n"), "");
    EXPECT_EQ(read_all(p), "first\n");
    EXPECT_EQ(adai::write_file_atomically(p.string(), "second, longer\n"), "");
    EXPECT_EQ(read_all(p), "second, longer\n");
    EXPECT_EQ(adai::write_file_atomically(p.string(), ""), "");
    EXPECT_EQ(read_all(p), "");
}

// The TD-495 case: a write that fails part-way (here a 4 KiB file-size limit, as disk full would)
// must leave the old file whole and no temporary behind. Run in a child so the limit applies only
// there.
TEST_F(AtomicFileTest, FailedWriteLeavesTheOldFileWhole) {
    const fs::path p = dir_ / "queue.txt";
    const std::string old_content(20000, 'a');
    ASSERT_EQ(adai::write_file_atomically(p.string(), old_content), "");

    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        std::signal(SIGXFSZ, SIG_IGN);  // the write fails with EFBIG instead of killing this child
        rlimit limit{4096, 4096};
        ::setrlimit(RLIMIT_FSIZE, &limit);
        const std::string error = adai::write_file_atomically(p.string(), std::string(50000, 'b'));
        ::_exit(error.empty() ? 1 : 0);  // 0 = it reported the failure
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0) << "the failed write must be reported";
    EXPECT_EQ(read_all(p), old_content) << "the old file is untouched";
    EXPECT_TRUE(only_file_is("queue.txt")) << "no temporary file left behind";
}

TEST_F(AtomicFileTest, UnwritableDirectoryIsReportedAndTheFileKept) {
    if (::geteuid() == 0)
        GTEST_SKIP() << "root ignores directory permissions";
    const fs::path p = dir_ / "queue.txt";
    ASSERT_EQ(adai::write_file_atomically(p.string(), "kept\n"), "");
    fs::permissions(dir_, fs::perms::owner_read | fs::perms::owner_exec);
    const std::string error = adai::write_file_atomically(p.string(), "lost\n");
    fs::permissions(dir_, fs::perms::owner_all);
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(read_all(p), "kept\n");
}
