#ifndef BTAG_SETTINGS_H
#define BTAG_SETTINGS_H

#pragma once

#include <array>
#include <string>
#include <string_view>
#include <cmath>
#include <algorithm>
#include <vector>

// Canonical UParTAK4 working-point order, from loosest to tightest.  Keep this
// definition shared by efficiency production and runtime SF application.
inline constexpr std::array<std::string_view, 5> kBTagInclusiveWorkingPoints = {
    "L", "M", "T", "XT", "XXT"
};

// Mutually exclusive observed categories corresponding to the ordered WPs.
inline constexpr std::array<std::string_view, 6> kBTagExclusiveCategories = {
    "N", "LnotM", "MnotT", "TnotXT", "XTnotXXT", "XXT"
};

inline double bTagMaxAbsEta(const std::string &year) {
    return (year == "2016preVFP" || year == "2016postVFP") ? 2.4 : 2.5;
}

// The Run-2 UParTAK4 SF payloads are tabulated below |eta|=2.4, while the
// selected-jet acceptance remains |eta|<2.5 for 2017/2018 and 2024/2025.
// Clamp only the SF lookup coordinate to the valid payload domain; do not
// discard those jets.
inline double bTagSFAbsEta(const std::string &year, double eta) {
    const double payload_max = (year == "2024Prompt" || year == "2025") ? 2.5 : 2.4;
    return std::min(std::abs(eta), std::nextafter(payload_max, 0.0));
}

inline std::string bTagSafeYearToken(std::string year) {
    if (year == "2022Re-recoBCD") return "2022Re_recoBCD";
    if (year == "2022Re-recoE+PromptFG") return "2022Re_recoE_PromptFG";
    return year;
}

std::vector<std::string> bTagWorkingPointsForChannel(const std::string &channel);
bool bTagScaleFactorsEnabled(const std::string &channel);

#endif
