#include <catch2/catch.hpp>
#include "core/utils.hpp"
//
//
TEST_CASE("Test fill_random for floats", "[utils][fill_random]") {
    std::vector<float> vec (100, 0);
    fill_random(vec.data(), vec.size());
    for (const auto val: vec) {
        REQUIRE(val >= 0.0);
        REQUIRE(val <= 1.0);
    }

    stats stats_(vec);
    REQUIRE(stats_.mean() > 0.0);
    REQUIRE(stats_.mean() < 1.0);
    REQUIRE(stats_.variance() > 0.0);
    REQUIRE(stats_.stdev() > 0.0);
}
//
//
TEST_CASE("Test fill_random for unsigned chars", "[utils][fill_random]") {
    std::vector<unsigned char> vec (100, 0);
    fill_random(vec.data(), vec.size());
    float sum(0);
    for (const auto val: vec) {
        REQUIRE(val >= 0);
        REQUIRE(val <= 255);
        sum += static_cast<float>(val);
    }
    
    std::sort(vec.begin(), vec.end());
    REQUIRE(vec.back() > 0);
    REQUIRE(sum > 0);
    
}
//
//
TEST_CASE("Testing to_string functions", "[utils][S]") {
    REQUIRE(S(true) == "true");
    REQUIRE(S(false) == "false");
    REQUIRE(S(1) == "1");
    REQUIRE(S(-10) == "-10");
    SECTION("Floating point numebrs") {
        std::string s = S(2.343);
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        REQUIRE( s == "2.343");
    }
}
//
//
TEST_CASE("Testing from_string method", "[utils][from_string]") {
    SECTION("Test true values") {
        std::vector<std::string> trues = {"1", "on", "yes", "true", "ON", "Yes", "YES", "TrUe"};
        for (const auto val: trues) {
            REQUIRE(from_string<bool>(val.c_str()) == true);
        }
    }
    SECTION("Test false values") {
        std::vector<std::string> trues = {"0", "off", "no", "false", "OFF", "No", "NO", "FaLsE"};
        for (const auto val: trues) {
            REQUIRE(from_string<bool>(val.c_str()) == false);
        }
    }
    REQUIRE(from_string<int>("10") == 10);
    REQUIRE(from_string<int>("-1123") == -1123);
    REQUIRE(from_string<std::string>("Hello World!") == std::string("Hello World!"));
}
//
//
TEST_CASE("Test format function", "[utils][format][fmt]") {
    REQUIRE(fmt("Hello %s!", "world") == std::string("Hello world!"));
    REQUIRE(fmt("Year %d.", 2018) == std::string("Year 2018."));
    REQUIRE(fmt("True is %s", "true") == std::string("True is true"));
    REQUIRE(fmt("%s is False", "false") == std::string("false is False"));
    REQUIRE(fmt("%s %.3f %d %s/%s", "hello", 0.0523, 34, "true", "false") == std::string("hello 0.052 34 true/false"));
}
//
//
TEST_CASE("Test sharded_vector class", "[utils][sharded_vector][shard]") {
    std::vector<int> vec(1000, 0);

    SECTION("1  1") {
        sharded_vector<int> sv(vec, 1, 0);
        REQUIRE(sv.size() == 1000);
        REQUIRE(sv.num_shards() == 1);
        REQUIRE(sv.my_shard() == 0);
        REQUIRE(sv.shard_begin() == 0);
        REQUIRE(sv.shard_length() == 1000);
    }
    SECTION("3  3") {
        for (int i=0; i<3; ++i) {
            sharded_vector<int> sv(vec, 3, i);
            REQUIRE(sv.size() == 1000);
            REQUIRE(sv.num_shards() == 3);
            REQUIRE(sv.my_shard() == i);
            REQUIRE(sv.shard_begin() == (i==0?0:(i==1?334:667)));
            REQUIRE(sv.shard_length() == (i==0?334:333));
        }
    }
}
//
//
TEST_CASE("Testing running_average", "[utils][average][running_average]") {
    std::vector<float> vec = {0.832895957, 0.541447969, 0.716991421, 0.620069652, 0.69472857,
                              0.581988332, 0.562546247, 0.362399979, 0.906282029, 0.646835666};
    std::vector<float> averages = {0.832895957, 0.687171963, 0.697111782, 0.67785125, 0.681226714,
                                   0.664686983, 0.65009545, 0.614133516, 0.646594462, 0.646618582};

    running_average ra;
    const double epsilon = 10e-7;
    for (int i=0; i<10; ++i) {
        ra.update(vec[i]);
        REQUIRE(ra.num_steps() == (i+1));
        REQUIRE(std::fabs(ra.value() - averages[i]) <= epsilon);
    }
}
//
//
TEST_CASE("The stats module for positive numbers", "[utils][stats]") {
    
    const float epsilon = 10e-6;
    std::vector<float> vec = {0.832895957, 0.541447969, 0.716991421, 0.620069652, 0.69472857,
                              0.581988332, 0.562546247, 0.362399979, 0.906282029, 0.646835666};

    stats stats_(vec);

    SECTION("Mean") {
        REQUIRE(std::fabs(stats_.mean() - 0.646618582) <= epsilon);
    }
    SECTION("Variance") {
        REQUIRE(std::fabs(stats_.variance() - 0.023686941) <= epsilon);
    }
    SECTION("Standard deviation") {
        REQUIRE(std::fabs(stats_.stdev() - 0.153905622) <= epsilon);
    }
    SECTION("Min") {
        REQUIRE(std::fabs(stats_.min() - 0.362399979) <= epsilon);
    }
    SECTION("Max") {
        REQUIRE(std::fabs(stats_.max() - 0.906282029) <= epsilon);
    }
}
//
//