// Unit and integration tests for MNS Manager GUI code.
//
// Part 1: Pure unit tests for MnsJsonHelpers (no server needed).
// Part 2: Live integration tests against a running mns_server (auto-skip when
//          unreachable, same pattern as model_name_service_live_test.cpp).
//
// Configuration (env vars, both optional):
//   MNS_SERVER_HOST   default: localhost
//   MNS_SERVER_PORT   default: 8083

#include <gtest/gtest.h>

#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include "MnsJsonHelpers.hpp"

using mns_gui::json_array_objects;
using mns_gui::json_escape;
using mns_gui::json_pretty;
using mns_gui::json_value;
using mns_gui::ParsedUrl;

// ============================================================================
// Part 1: Pure unit tests — JSON helpers and URL parser
// ============================================================================

// ---------------------------------------------------------------------------
// json_value
// ---------------------------------------------------------------------------

TEST(MnsJsonValue, ExtractsStringValue) {
    std::string body = R"({"model_name":"chatbot-v3","state":"training"})";
    EXPECT_EQ("chatbot-v3", json_value(body, "model_name"));
    EXPECT_EQ("training", json_value(body, "state"));
}

TEST(MnsJsonValue, ExtractsNumericValue) {
    std::string body = R"({"d_model":128,"num_heads":4})";
    EXPECT_EQ("128", json_value(body, "d_model"));
    EXPECT_EQ("4", json_value(body, "num_heads"));
}

TEST(MnsJsonValue, ExtractsBooleanValue) {
    std::string body = R"({"trained":true,"pending":false})";
    EXPECT_EQ("true", json_value(body, "trained"));
    EXPECT_EQ("false", json_value(body, "pending"));
}

TEST(MnsJsonValue, ReturnsEmptyForMissingKey) {
    std::string body = R"({"name":"test"})";
    EXPECT_EQ("", json_value(body, "missing"));
}

TEST(MnsJsonValue, HandlesSpacesAfterColon) {
    std::string body = R"({"key": "value"})";
    EXPECT_EQ("value", json_value(body, "key"));
}

TEST(MnsJsonValue, HandlesEmptyStringValue) {
    std::string body = R"({"run_id":""})";
    EXPECT_EQ("", json_value(body, "run_id"));
}

TEST(MnsJsonValue, HandlesEmptyBody) {
    EXPECT_EQ("", json_value("", "key"));
    EXPECT_EQ("", json_value("{}", "key"));
}

TEST(MnsJsonValue, HandlesNestedObject) {
    std::string body = R"({"artifact":{"host":"server1","path":"/tmp/m.bin"},"state":"candidate"})";
    EXPECT_EQ("candidate", json_value(body, "state"));
    EXPECT_EQ("server1", json_value(body, "host"));
    EXPECT_EQ("/tmp/m.bin", json_value(body, "path"));
}

TEST(MnsJsonValue, DoesNotMatchPartialKeys) {
    std::string body = R"({"model_name":"test","name":"other"})";
    EXPECT_EQ("other", json_value(body, "name"));
    EXPECT_EQ("test", json_value(body, "model_name"));
}

TEST(MnsJsonValue, HandlesNullValue) {
    std::string body = R"({"retired":null,"name":"test"})";
    EXPECT_EQ("null", json_value(body, "retired"));
}

// ---------------------------------------------------------------------------
// json_array_objects
// ---------------------------------------------------------------------------

TEST(MnsJsonArrayObjects, ExtractsObjectsFromNamedArray) {
    std::string body = R"({"models":[{"name":"a"},{"name":"b"},{"name":"c"}]})";
    auto objs = json_array_objects(body, "models");
    ASSERT_EQ(3u, objs.size());
    EXPECT_EQ("a", json_value(objs[0], "name"));
    EXPECT_EQ("b", json_value(objs[1], "name"));
    EXPECT_EQ("c", json_value(objs[2], "name"));
}

TEST(MnsJsonArrayObjects, HandlesEmptyArray) {
    std::string body = R"({"models":[]})";
    auto objs = json_array_objects(body, "models");
    EXPECT_TRUE(objs.empty());
}

TEST(MnsJsonArrayObjects, HandlesNestedObjects) {
    std::string body = R"({"models":[{"name":"a","artifact":{"host":"h1"}},{"name":"b"}]})";
    auto objs = json_array_objects(body, "models");
    ASSERT_EQ(2u, objs.size());
    EXPECT_EQ("h1", json_value(objs[0], "host"));
    EXPECT_EQ("b", json_value(objs[1], "name"));
}

TEST(MnsJsonArrayObjects, ReturnEmptyForMissingKey) {
    std::string body = R"({"roles":[]})";
    auto objs = json_array_objects(body, "models");
    EXPECT_TRUE(objs.empty());
}

TEST(MnsJsonArrayObjects, HandlesSingleObject) {
    std::string body = R"({"roles":[{"role":"chatbot","production_model":"v3"}]})";
    auto objs = json_array_objects(body, "roles");
    ASSERT_EQ(1u, objs.size());
    EXPECT_EQ("chatbot", json_value(objs[0], "role"));
    EXPECT_EQ("v3", json_value(objs[0], "production_model"));
}

TEST(MnsJsonArrayObjects, HandlesNoKeyTopLevelArray) {
    std::string body = R"([{"a":1},{"b":2}])";
    auto objs = json_array_objects(body, "nonexistent");
    ASSERT_EQ(2u, objs.size());
}

// ---------------------------------------------------------------------------
// json_escape
// ---------------------------------------------------------------------------

TEST(MnsJsonEscape, EscapesQuotesAndBackslash) {
    EXPECT_EQ(R"(hello \"world\")", json_escape("hello \"world\""));
    EXPECT_EQ(R"(path\\to\\file)", json_escape("path\\to\\file"));
}

TEST(MnsJsonEscape, EscapesNewline) {
    EXPECT_EQ("line1\\nline2", json_escape("line1\nline2"));
}

TEST(MnsJsonEscape, PassthroughPlainString) {
    EXPECT_EQ("simple", json_escape("simple"));
    EXPECT_EQ("", json_escape(""));
}

TEST(MnsJsonEscape, HandlesAllSpecialChars) {
    std::string input = "a\"b\\c\nd";
    std::string expected = "a\\\"b\\\\c\\nd";
    EXPECT_EQ(expected, json_escape(input));
}

// ---------------------------------------------------------------------------
// json_pretty
// ---------------------------------------------------------------------------

TEST(MnsJsonPretty, IndentsSimpleObject) {
    std::string input = R"({"a":"b","c":1})";
    std::string result = json_pretty(input);
    EXPECT_NE(result.find('\n'), std::string::npos);
    EXPECT_NE(result.find("  "), std::string::npos);
    EXPECT_NE(result.find("\"a\""), std::string::npos);
}

TEST(MnsJsonPretty, HandlesNestedBraces) {
    std::string input = R"({"a":{"b":"c"}})";
    std::string result = json_pretty(input);
    // Should have multiple levels of indentation
    EXPECT_NE(result.find("    "), std::string::npos);
}

TEST(MnsJsonPretty, PreservesStringContent) {
    std::string input = R"({"path":"/tmp/a,b{c}"})";
    std::string result = json_pretty(input);
    EXPECT_NE(result.find("/tmp/a,b{c}"), std::string::npos);
}

TEST(MnsJsonPretty, HandlesEmptyObject) {
    std::string result = json_pretty("{}");
    EXPECT_NE(result.find('{'), std::string::npos);
    EXPECT_NE(result.find('}'), std::string::npos);
}

TEST(MnsJsonPretty, HandlesArray) {
    std::string input = R"({"items":[1,2,3]})";
    std::string result = json_pretty(input);
    EXPECT_NE(result.find('['), std::string::npos);
}

// ---------------------------------------------------------------------------
// ParsedUrl
// ---------------------------------------------------------------------------

TEST(MnsParsedUrl, ParsesHostAndPort) {
    auto p = ParsedUrl::from("http://myhost:9090");
    EXPECT_EQ("myhost", p.host);
    EXPECT_EQ(9090, p.port);
}

TEST(MnsParsedUrl, ParsesHostOnly) {
    auto p = ParsedUrl::from("http://myhost");
    EXPECT_EQ("myhost", p.host);
    EXPECT_EQ(8083, p.port);
}

TEST(MnsParsedUrl, StripsTrailingPath) {
    auto p = ParsedUrl::from("http://server:8083/some/path");
    EXPECT_EQ("server", p.host);
    EXPECT_EQ(8083, p.port);
}

TEST(MnsParsedUrl, HandlesHttps) {
    auto p = ParsedUrl::from("https://secure:4443");
    EXPECT_EQ("secure", p.host);
    EXPECT_EQ(4443, p.port);
}

TEST(MnsParsedUrl, HandlesNoScheme) {
    auto p = ParsedUrl::from("bare-host:7777");
    EXPECT_EQ("bare-host", p.host);
    EXPECT_EQ(7777, p.port);
}

TEST(MnsParsedUrl, DefaultsLocalhostPort) {
    auto p = ParsedUrl::from("");
    EXPECT_EQ("", p.host);
    EXPECT_EQ(8083, p.port);
}

TEST(MnsParsedUrl, ParsesLocalhost) {
    auto p = ParsedUrl::from("http://localhost:8083");
    EXPECT_EQ("localhost", p.host);
    EXPECT_EQ(8083, p.port);
}

TEST(MnsParsedUrl, HandlesIPAddress) {
    auto p = ParsedUrl::from("http://192.168.1.19:8083");
    EXPECT_EQ("192.168.1.19", p.host);
    EXPECT_EQ(8083, p.port);
}

// ============================================================================
// Part 2: Live integration tests — require a running mns_server
// ============================================================================

namespace {

std::string live_host() {
    const char* env = std::getenv("MNS_SERVER_HOST");
    return env ? std::string(env) : "localhost";
}

int live_port() {
    const char* env = std::getenv("MNS_SERVER_PORT");
    return env ? std::atoi(env) : 8083;
}

httplib::Client make_live_client() {
    httplib::Client c(live_host(), live_port());
    c.set_connection_timeout(std::chrono::seconds(5));
    c.set_read_timeout(std::chrono::seconds(5));
    return c;
}

bool live_server_reachable() {
    auto c = make_live_client();
    auto res = c.Get("/health");
    return res && res->status == 200;
}

std::string unique_model(const std::string& tag = "") {
    static std::atomic<int> counter{0};
    std::ostringstream oss;
    oss << "gui" << static_cast<int>(::getpid()) << "t" << static_cast<long>(std::time(nullptr))
        << "c" << counter.fetch_add(1);
    if (!tag.empty())
        oss << "-" << tag;
    return oss.str();
}

std::string unique_role(const std::string& tag = "") {
    static std::atomic<int> counter{0};
    std::ostringstream oss;
    oss << "guirole" << static_cast<int>(::getpid()) << "c" << counter.fetch_add(1);
    if (!tag.empty())
        oss << "-" << tag;
    return oss.str();
}

std::string register_body(const std::string& name, const std::string& role = "") {
    std::ostringstream body;
    body << "{\"model_name\":\"" << name << "\"";
    if (!role.empty())
        body << ",\"role\":\"" << role << "\"";
    body << ",\"arch\":{\"d_model\":64,\"num_heads\":2,\"d_ff\":128"
            ",\"num_encoder_layers\":1,\"num_decoder_layers\":1,\"max_seq_length\":64}}";
    return body.str();
}

}  // namespace

class MnsManagerGUILiveTest : public ::testing::Test {
   protected:
    void SetUp() override {
        if (!live_server_reachable()) {
            GTEST_SKIP() << "mns_server not reachable at " << live_host() << ":" << live_port();
        }
    }
};

// ---------------------------------------------------------------------------
// List models and parse the response with GUI helpers
// ---------------------------------------------------------------------------

TEST_F(MnsManagerGUILiveTest, ListModels_ParsesWithJsonArrayObjects) {
    auto c = make_live_client();
    const auto name = unique_model("list");
    c.Post("/models", register_body(name), "application/json");

    auto res = c.Get("/models");
    ASSERT_TRUE(res);
    ASSERT_EQ(200, res->status);

    auto objs = json_array_objects(res->body, "models");
    EXPECT_GE(objs.size(), 1u);

    bool found = false;
    for (const auto& obj : objs) {
        if (json_value(obj, "model_name") == name) {
            found = true;
            EXPECT_EQ("initializing", json_value(obj, "state"));
        }
    }
    EXPECT_TRUE(found) << "Registered model not found in list response";
}

// ---------------------------------------------------------------------------
// List roles and parse the response with GUI helpers
// ---------------------------------------------------------------------------

TEST_F(MnsManagerGUILiveTest, ListRoles_ParsesWithJsonArrayObjects) {
    auto c = make_live_client();
    const auto role = unique_role("roles");
    const auto name = unique_model("roles");
    c.Post("/models", register_body(name, role), "application/json");

    auto res = c.Get("/roles");
    ASSERT_TRUE(res);
    ASSERT_EQ(200, res->status);

    auto objs = json_array_objects(res->body, "roles");
    bool found = false;
    for (const auto& obj : objs) {
        if (json_value(obj, "role") == role) {
            found = true;
        }
    }
    EXPECT_TRUE(found) << "Role '" << role << "' not found in /roles response";
}

// ---------------------------------------------------------------------------
// Get model detail and parse with json_value
// ---------------------------------------------------------------------------

TEST_F(MnsManagerGUILiveTest, GetModel_ParsesFullRecord) {
    auto c = make_live_client();
    const auto name = unique_model("detail");
    const auto role = unique_role("detail");
    c.Post("/models", register_body(name, role), "application/json");

    auto res = c.Get("/models/" + name);
    ASSERT_TRUE(res);
    ASSERT_EQ(200, res->status);

    EXPECT_EQ(name, json_value(res->body, "model_name"));
    EXPECT_EQ(role, json_value(res->body, "role"));
    EXPECT_EQ("initializing", json_value(res->body, "state"));
    EXPECT_FALSE(json_value(res->body, "model_id").empty());
    EXPECT_FALSE(json_value(res->body, "created_utc").empty());
    EXPECT_EQ("64", json_value(res->body, "d_model"));
}

// ---------------------------------------------------------------------------
// Full lifecycle: register -> training -> candidate -> promote
// parsed with GUI helpers at each step
// ---------------------------------------------------------------------------

TEST_F(MnsManagerGUILiveTest, FullLifecycle_RegisterTrainPromote) {
    auto c = make_live_client();
    const auto name = unique_model("lifecycle");
    const auto role = unique_role("lifecycle");

    // Register
    auto r1 = c.Post("/models", register_body(name, role), "application/json");
    ASSERT_TRUE(r1);
    ASSERT_EQ(201, r1->status);
    EXPECT_EQ("initializing", json_value(r1->body, "state"));

    // Set training
    std::string train_body = "{\"state\":\"training\",\"run_id\":\"gui-run-1\"}";
    auto r2 = c.Put("/models/" + name + "/state", train_body, "application/json");
    ASSERT_TRUE(r2);
    EXPECT_EQ(200, r2->status);
    EXPECT_EQ("training", json_value(r2->body, "state"));

    // Set candidate
    std::string cand_body =
        "{\"state\":\"candidate\",\"run_id\":\"gui-run-1\""
        ",\"artifact\":{\"host\":\"\",\"path\":\"/tmp/test.bin\""
        ",\"checksum\":\"abc\",\"format\":\"adai-native\"}}";
    auto r3 = c.Put("/models/" + name + "/state", cand_body, "application/json");
    ASSERT_TRUE(r3);
    EXPECT_EQ(200, r3->status);
    EXPECT_EQ("candidate", json_value(r3->body, "state"));
    EXPECT_EQ("/tmp/test.bin", json_value(r3->body, "path"));

    // Promote
    std::string prom_body = "{\"model_name\":\"" + name + "\"}";
    auto r4 = c.Put("/roles/" + role + "/production", prom_body, "application/json");
    ASSERT_TRUE(r4);
    EXPECT_EQ(200, r4->status);
    EXPECT_EQ(name, json_value(r4->body, "promoted"));

    // Verify via roles list
    auto r5 = c.Get("/roles");
    ASSERT_TRUE(r5);
    auto roles_objs = json_array_objects(r5->body, "roles");
    bool found = false;
    for (const auto& obj : roles_objs) {
        if (json_value(obj, "role") == role) {
            EXPECT_EQ(name, json_value(obj, "production_model"));
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

// ---------------------------------------------------------------------------
// Retire and delete parsed with GUI helpers
// ---------------------------------------------------------------------------

TEST_F(MnsManagerGUILiveTest, RetireAndDelete) {
    auto c = make_live_client();
    const auto name = unique_model("retire-del");
    c.Post("/models", register_body(name), "application/json");

    // Initializing -> can't delete training/candidate/production
    // but initializing CAN be deleted directly
    auto r1 = c.Delete("/models/" + name);
    ASSERT_TRUE(r1);
    EXPECT_EQ(200, r1->status);

    // Register again, move through lifecycle, then retire and delete
    const auto name2 = unique_model("retire-del2");
    c.Post("/models", register_body(name2), "application/json");
    c.Put("/models/" + name2 + "/state", "{\"state\":\"training\",\"run_id\":\"r1\"}",
          "application/json");
    c.Put("/models/" + name2 + "/state", "{\"state\":\"candidate\",\"run_id\":\"r1\"}",
          "application/json");

    // Candidate can't be deleted directly
    auto r2 = c.Delete("/models/" + name2);
    ASSERT_TRUE(r2);
    EXPECT_EQ(409, r2->status);

    // Retire first
    auto r3 = c.Put("/models/" + name2 + "/state", "{\"state\":\"retired\"}", "application/json");
    ASSERT_TRUE(r3);
    EXPECT_EQ(200, r3->status);
    EXPECT_EQ("retired", json_value(r3->body, "state"));

    // Now delete
    auto r4 = c.Delete("/models/" + name2);
    ASSERT_TRUE(r4);
    EXPECT_EQ(200, r4->status);

    // Confirm gone
    auto r5 = c.Get("/models/" + name2);
    ASSERT_TRUE(r5);
    EXPECT_EQ(404, r5->status);
}

// ---------------------------------------------------------------------------
// State filter in list (models table uses this)
// ---------------------------------------------------------------------------

TEST_F(MnsManagerGUILiveTest, ListModels_StateFilter) {
    auto c = make_live_client();
    const auto name = unique_model("filter");
    c.Post("/models", register_body(name), "application/json");
    c.Put("/models/" + name + "/state", "{\"state\":\"training\",\"run_id\":\"f1\"}",
          "application/json");

    // Filter by training — should include this model
    auto r1 = c.Get("/models?state=training");
    ASSERT_TRUE(r1);
    EXPECT_EQ(200, r1->status);
    auto objs_training = json_array_objects(r1->body, "models");
    bool found_training = false;
    for (const auto& obj : objs_training) {
        if (json_value(obj, "model_name") == name)
            found_training = true;
        EXPECT_EQ("training", json_value(obj, "state"));
    }
    EXPECT_TRUE(found_training);

    // Filter by production — should NOT include this model
    auto r2 = c.Get("/models?state=production");
    ASSERT_TRUE(r2);
    auto objs_prod = json_array_objects(r2->body, "models");
    for (const auto& obj : objs_prod) {
        EXPECT_NE(name, json_value(obj, "model_name"));
    }
}

// ---------------------------------------------------------------------------
// json_pretty on live server response
// ---------------------------------------------------------------------------

TEST_F(MnsManagerGUILiveTest, PrettyPrint_LiveResponse) {
    auto c = make_live_client();
    const auto name = unique_model("pretty");
    c.Post("/models", register_body(name), "application/json");

    auto res = c.Get("/models/" + name);
    ASSERT_TRUE(res);
    ASSERT_EQ(200, res->status);

    std::string pretty = json_pretty(res->body);
    EXPECT_GT(pretty.size(), res->body.size());
    EXPECT_NE(pretty.find('\n'), std::string::npos);
    EXPECT_NE(pretty.find(name), std::string::npos);
}

// ---------------------------------------------------------------------------
// Role filter in list (models table uses this)
// ---------------------------------------------------------------------------

TEST_F(MnsManagerGUILiveTest, ListModels_RoleFilter) {
    auto c = make_live_client();
    const auto role = unique_role("rolefilter");
    const auto name = unique_model("rolefilter");
    c.Post("/models", register_body(name, role), "application/json");

    auto res = c.Get("/models?role=" + role);
    ASSERT_TRUE(res);
    EXPECT_EQ(200, res->status);
    auto objs = json_array_objects(res->body, "models");
    ASSERT_EQ(1u, objs.size());
    EXPECT_EQ(name, json_value(objs[0], "model_name"));
}
