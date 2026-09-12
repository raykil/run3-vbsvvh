#include "btag_settings.h"

#include <fstream>
#include <algorithm>
#include <map>
#include <stdexcept>

namespace {

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t");
    return value.substr(begin, end - begin + 1);
}

const std::map<std::string, std::vector<std::string>> &applyBTagConfig() {
    static const auto config = [] {
        constexpr const char *path = "applybtag.yaml";
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot read b-tag application configuration " + std::string(path));

        std::map<std::string, std::vector<std::string>> parsed;
        std::string line;
        while (std::getline(input, line)) {
            const auto comment = line.find('#');
            if (comment != std::string::npos) line.erase(comment);
            line = trim(line);
            if (line.empty()) continue;

            const auto separator = line.find(':');
            if (separator == std::string::npos || line.find(':', separator + 1) != std::string::npos)
                throw std::runtime_error("Invalid applybtag.yaml entry: " + line);
            const auto channel = trim(line.substr(0, separator));
            auto value = trim(line.substr(separator + 1));
            if (channel.empty() || value.size() < 2 || value.front() != '[' || value.back() != ']')
                throw std::runtime_error("Invalid applybtag.yaml entry: " + line);
            value = trim(value.substr(1, value.size() - 2));
            std::vector<std::string> working_points;
            if (!value.empty()) {
                std::size_t start = 0;
                while (start <= value.size()) {
                    const auto comma = value.find(',', start);
                    auto wp = trim(value.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
                    if (std::find(kBTagInclusiveWorkingPoints.begin(), kBTagInclusiveWorkingPoints.end(), wp) == kBTagInclusiveWorkingPoints.end())
                        throw std::runtime_error("Unknown working point in applybtag.yaml entry: " + wp);
                    working_points.push_back(wp);
                    if (comma == std::string::npos) break;
                    start = comma + 1;
                }
            }
            if (std::adjacent_find(working_points.begin(), working_points.end()) != working_points.end())
                throw std::runtime_error("Duplicate working point in applybtag.yaml entry: " + line);
            for (std::size_t i = 1; i < working_points.size(); ++i)
                if (std::find(kBTagInclusiveWorkingPoints.begin(), kBTagInclusiveWorkingPoints.end(), working_points[i - 1]) >=
                    std::find(kBTagInclusiveWorkingPoints.begin(), kBTagInclusiveWorkingPoints.end(), working_points[i]))
                    throw std::runtime_error("Working points must be ordered L,M,T,XT,XXT: " + line);
            if (!parsed.emplace(channel, std::move(working_points)).second)
                throw std::runtime_error("Duplicate channel in applybtag.yaml: " + channel);
        }
        if (parsed.empty()) throw std::runtime_error("applybtag.yaml contains no channels");
        return parsed;
    }();
    return config;
}

} // namespace

std::vector<std::string> bTagWorkingPointsForChannel(const std::string &channel) {
    const auto &config = applyBTagConfig();
    const auto entry = config.find(channel);
    if (entry == config.end())
        throw std::runtime_error("Channel " + channel + " is missing from applybtag.yaml");
    return entry->second;
}

bool bTagScaleFactorsEnabled(const std::string &channel) {
    return !bTagWorkingPointsForChannel(channel).empty();
}
