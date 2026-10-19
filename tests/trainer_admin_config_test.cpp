/**
 * @file trainer_admin_config_test.cpp
 * @brief Tests for parse_trainer_admin_config() (TD-519), the PUT /admin/config parser shared by
 *        TrainerAdminAPI and TrainerServiceProxy.
 */

#include <gtest/gtest.h>
#include "TrainerAdminConfig.hpp"

using adai::parse_trainer_admin_config;

TEST(TrainerAdminConfigTest, AcceptsCompactSpacedAndPrettyPrintedBooleans) {
    for (const std::string body : {"{\"auto_save_enabled\":true}", "{\"auto_save_enabled\": true}",
                                   "{\n  \"auto_save_enabled\" :\ttrue\n}"}) {
        const auto r = parse_trainer_admin_config(body);
        EXPECT_TRUE(r.error.empty()) << body << ": " << r.error;
        ASSERT_TRUE(r.update.auto_save_enabled.has_value()) << body;
        EXPECT_TRUE(*r.update.auto_save_enabled) << body;
    }
    const auto off = parse_trainer_admin_config("{\"auto_save_enabled\": false}");
    ASSERT_TRUE(off.update.auto_save_enabled.has_value());
    EXPECT_FALSE(*off.update.auto_save_enabled);
}

TEST(TrainerAdminConfigTest, ReadsOnlyTheKeysSent) {
    const auto r = parse_trainer_admin_config(
        "{\"auto_save_every_samples\": 0, \"max_sessions_to_keep\": 3, \"unrelated\": \"x\"}");
    EXPECT_TRUE(r.error.empty()) << r.error;
    EXPECT_FALSE(r.update.auto_save_enabled.has_value());
    EXPECT_EQ(r.update.auto_save_every_samples, 0);
    EXPECT_FALSE(r.update.auto_save_every_minutes.has_value());
    EXPECT_EQ(r.update.max_sessions_to_keep, 3);
}

TEST(TrainerAdminConfigTest, RejectsWrongTypesRangesAndMalformedBodies) {
    for (const std::string body :
         {"", "not json", "[]", "{\"auto_save_enabled\":false", "{}", "{\"other\":1}",
          "{\"auto_save_enabled\":\"true\"}", "{\"auto_save_enabled\":1}",
          "{\"auto_save_enabled\":null}", "{\"auto_save_every_samples\":\"5\"}",
          "{\"auto_save_every_samples\":1.5}", "{\"auto_save_every_samples\":-1}",
          "{\"auto_save_every_minutes\":-1}", "{\"max_sessions_to_keep\":0}",
          "{\"max_sessions_to_keep\":4294967296}", "{\"port\":8084}", "{\"dir\":\"/tmp\"}",
          "{\"auto_save_enabled\":true,\"max_sessions_to_keep\":0}"}) {
        EXPECT_FALSE(parse_trainer_admin_config(body).error.empty()) << body;
    }
}
