/**
 * DataTransportFtpsTests — TD-339: DataTransport authenticates the registry's FTPS server.
 *
 * Live tests: a real FtpDataServer (FTPS, BUILD_FTPS) on loopback serves files to a real
 * DataTransport (libcurl, BUILD_FTP_TRANSPORT). Before TD-339 fetch() turned certificate
 * verification off unconditionally, so any server that spoke TLS was accepted.
 */
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "../src/DataTransport.hpp"
#include "../src/DatasetRegistry.hpp"
#include "../src/FtpDataServer.hpp"

#ifndef _WIN32

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

const std::string kPayload = "{\"input\":\"hello\",\"output\":\"world\"}\n";

class FtpsVerificationTest : public ::testing::Test {
   protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / "adai_dt_ftps_test";
        fs::remove_all(root_);
        data_dir_ = root_ / "data";
        download_dir_ = root_ / "downloads";
        fs::create_directories(data_dir_);
        std::ofstream(data_dir_ / "pairs.jsonl") << kPayload;
    }
    void TearDown() override {
        if (server_)
            server_->stop();
        fs::remove_all(root_);
    }

    // Starts an FTPS (or plain FTP) server; empty cert/key = a generated self-signed certificate.
    void start_server(int port, bool ftps, const std::string& cert = "",
                      const std::string& key = "") {
        server_ = std::make_unique<FtpDataServer>(data_dir_.string(), port, port + 100,
                                                  port + 103, "127.0.0.1", "", ftps, cert, key);
        server_->start();
        std::this_thread::sleep_for(300ms);
        port_ = port;
    }

    FileToken token() {
        const auto issued = server_->issue_token("run-td339", "pairs.jsonl", 5);
        FileToken tok;
        tok.registry_path = (data_dir_ / "pairs.jsonl").string();
        tok.ftp_path = "pairs.jsonl";
        tok.ftp_username = issued.username;
        tok.ftp_password = issued.password;
        tok.size_bytes = kPayload.size();
        tok.token_expires_utc = issued.token_expires_utc;
        return tok;
    }

    fs::path fetch(const FtpsVerification& verification, bool ftps) {
        DataTransport dt(verification);
        return dt.fetch(token(), "127.0.0.1", port_, download_dir_, 0, ftps);
    }

    // A self-signed certificate for 127.0.0.1 written by the openssl CLI; false if unavailable.
    bool make_certificate(const std::string& name, fs::path& cert, fs::path& key) {
        cert = root_ / (name + ".crt");
        key = root_ / (name + ".key");
        const std::string cmd = "openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj /CN=" +
                                name + " -addext subjectAltName=IP:127.0.0.1 -keyout " +
                                key.string() + " -out " + cert.string() + " >/dev/null 2>&1";
        return std::system(cmd.c_str()) == 0 && fs::exists(cert);
    }

    static std::string read_file(const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }

    fs::path root_, data_dir_, download_dir_;
    std::unique_ptr<FtpDataServer> server_;
    int port_ = 0;
};

}  // namespace

// The trainer's download step builds its FtpsVerification from this DatasetConfig.
TEST(FtpsVerificationConfigTest, ServiceConfigKeysReachDatasetConfig) {
    adai::ServiceConfig svc;
    svc.ftps_ca_file = "/etc/adai/registry-ca.pem";
    svc.ftps_pinned_public_key = "sha256//pin=";
    const DatasetConfig dcfg = DatasetRegistry::make_config(svc);
    EXPECT_EQ(dcfg.ftps_ca_file, "/etc/adai/registry-ca.pem");
    EXPECT_EQ(dcfg.ftps_pinned_public_key, "sha256//pin=");
}

TEST_F(FtpsVerificationTest, ServerPublishesItsPublicKeyPin) {
    start_server(21391, /*ftps=*/true);
    const std::string pin = server_->public_key_pin();
    EXPECT_EQ(pin.rfind("sha256//", 0), 0u) << pin;
    EXPECT_EQ(pin.size(), 8u + 44u) << pin;  // base64 of a 32-byte SHA-256
}

TEST_F(FtpsVerificationTest, MatchingPinDownloads) {
    start_server(21392, /*ftps=*/true);
    const fs::path got = fetch({"", server_->public_key_pin()}, /*ftps=*/true);
    EXPECT_EQ(read_file(got), kPayload);
}

// The impersonation case: a server with a different key. Before TD-339 this downloaded.
TEST_F(FtpsVerificationTest, DifferentPinIsRefused) {
    start_server(21393, /*ftps=*/true);
    const std::string other = "sha256//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
    EXPECT_THROW(fetch({"", other}, /*ftps=*/true), std::runtime_error);
    EXPECT_FALSE(fs::exists(DataTransport::local_path_for(token(), download_dir_)) &&
                 fs::file_size(DataTransport::local_path_for(token(), download_dir_)) > 0);
}

TEST_F(FtpsVerificationTest, CaFileMustSignTheServerCertificate) {
    fs::path cert, key, other_cert, other_key;
    if (!make_certificate("adai-registry", cert, key) ||
        !make_certificate("someone-else", other_cert, other_key))
        GTEST_SKIP() << "openssl CLI unavailable";
    start_server(21394, /*ftps=*/true, cert.string(), key.string());
    EXPECT_THROW(fetch({other_cert.string(), ""}, /*ftps=*/true), std::runtime_error);
    const fs::path got = fetch({cert.string(), ""}, /*ftps=*/true);
    EXPECT_EQ(read_file(got), kPayload);
}

TEST_F(FtpsVerificationTest, MissingCaFileFailsBeforeConnecting) {
    start_server(21395, /*ftps=*/true);
    try {
        fetch({(root_ / "missing.pem").string(), ""}, /*ftps=*/true);
        FAIL() << "expected a missing CA file to fail";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("FTPS_CA_FILE"), std::string::npos) << e.what();
    }
}

// With verification configured, a registry that offers plain FTP doesn't get the credentials.
TEST_F(FtpsVerificationTest, ConfiguredVerificationRefusesPlainFtp) {
    start_server(21396, /*ftps=*/false);
    try {
        fetch({"", "sha256//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}, /*ftps=*/false);
        FAIL() << "expected plain FTP to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("plain FTP"), std::string::npos) << e.what();
    }
}

// Unchanged for deployments that haven't configured verification yet (a warning is logged).
TEST_F(FtpsVerificationTest, UnverifiedFtpsStillDownloads) {
    start_server(21397, /*ftps=*/true);
    const fs::path got = fetch({}, /*ftps=*/true);
    EXPECT_EQ(read_file(got), kPayload);
}

#endif  // _WIN32
