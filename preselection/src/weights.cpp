#include "weights.h"
#include "btag_settings.h"
#include "corrections.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
constexpr double kBTagDenominatorEpsilon = 1.e-8;
std::atomic<unsigned long long> g_btag_negative_intermediate{0};
std::atomic<unsigned long long> g_btag_tiny_denominator{0};
std::atomic<unsigned long long> g_btag_invalid_probability{0};
std::mutex g_btag_diagnostic_mutex;
std::map<std::string, unsigned long long> g_btag_failure_details;

constexpr std::array<std::string_view, 29> kBTagHFSources = {
    "correlated", "uncorrelated", "statistic", "pileup", "isrdef", "fsrdef",
    "muf", "mur", "pdf", "as", "pdfas", "ttbar", "jes", "jer", "type3",
    "bfragmentation", "topmass", "hdamp",
    "jesRegrouped_Absolute", "jesRegrouped_Absolute_2024",
    "jesRegrouped_BBEC1", "jesRegrouped_BBEC1_2024",
    "jesRegrouped_EC2", "jesRegrouped_EC2_2024",
    "jesRegrouped_FlavorQCD", "jesRegrouped_HF", "jesRegrouped_HF_2024",
    "jesRegrouped_RelativeBal", "jesRegrouped_RelativeSample_2024"
};

const std::map<std::string, std::set<std::string>> kBTagHFAvailableSources = {
    {"2016preVFP", {"correlated", "uncorrelated", "statistic", "pileup", "isrdef", "fsrdef",
                     "muf", "mur", "pdf", "as", "ttbar"}},
    {"2016postVFP", {"correlated", "uncorrelated", "statistic", "pileup", "isrdef", "fsrdef",
                      "muf", "mur", "pdf", "as", "ttbar"}},
    {"2017", {"correlated", "uncorrelated", "statistic", "pileup", "isrdef", "fsrdef",
              "muf", "mur", "pdf", "as", "ttbar"}},
    {"2018", {"correlated", "uncorrelated", "statistic", "pileup", "isrdef", "fsrdef",
              "muf", "mur", "pdf", "as", "ttbar"}},
    {"2024Prompt", {"correlated", "uncorrelated", "statistic", "pileup", "isrdef", "fsrdef",
                     "muf", "mur", "pdfas", "jes", "jer", "type3", "bfragmentation",
                     "topmass", "hdamp"}},
    // The 2025 payload uses the same logical sources except that topmass is
    // labelled mass, while generic pdfas/jes/bfragmentation are absent.
    {"2025", {"correlated", "uncorrelated", "statistic", "pileup", "isrdef", "fsrdef",
               "muf", "mur", "pdf", "as", "ttbar", "jer", "type3", "topmass", "hdamp",
               "jesRegrouped_Absolute", "jesRegrouped_Absolute_2024",
               "jesRegrouped_BBEC1", "jesRegrouped_BBEC1_2024",
               "jesRegrouped_EC2", "jesRegrouped_EC2_2024",
               "jesRegrouped_FlavorQCD", "jesRegrouped_HF", "jesRegrouped_HF_2024",
               "jesRegrouped_RelativeBal", "jesRegrouped_RelativeSample_2024"}}
};

enum class BTagJESScheme { None, Total, Regrouped11 };

struct BTagJESMapping {
    std::string_view analysis_source;
    std::string_view payload_source;
};

constexpr std::array<BTagJESMapping, 11> kBTagJES2025Mapping = {{
    {"Absolute", "jesRegrouped_Absolute"},
    {"BBEC1", "jesRegrouped_BBEC1"},
    {"EC2", "jesRegrouped_EC2"},
    {"HF", "jesRegrouped_HF"},
    {"RelativeBal", "jesRegrouped_RelativeBal"},
    {"FlavorQCD", "jesRegrouped_FlavorQCD"},
    {"AbsoluteYear", "jesRegrouped_Absolute_2024"},
    {"BBEC1Year", "jesRegrouped_BBEC1_2024"},
    {"EC2Year", "jesRegrouped_EC2_2024"},
    {"HFYear", "jesRegrouped_HF_2024"},
    {"RelativeSampleYear", "jesRegrouped_RelativeSample_2024"},
}};

bool bTagHFSourceAvailable(const std::string &year, std::string_view source);

BTagJESScheme bTagJESSchemeForYear(std::string_view year) {
    if (year == "2025") return BTagJESScheme::Regrouped11;
    if (year == "2024Prompt" || bTagHFSourceAvailable(std::string(year), "jes"))
        return BTagJESScheme::Total;
    return BTagJESScheme::None;
}

bool bTagHFSourceIsRegroupedJES(std::string_view source) {
    return std::find_if(kBTagJES2025Mapping.begin(), kBTagJES2025Mapping.end(),
                        [source](const BTagJESMapping &mapping) {
                            return mapping.payload_source == source;
                        }) != kBTagJES2025Mapping.end();
}

bool bTagHFSourceAvailable(const std::string &year, std::string_view source) {
    const auto year_it = kBTagHFAvailableSources.find(year);
    return year_it != kBTagHFAvailableSources.end() && year_it->second.count(std::string(source));
}

// Keep public branch names logical and translate only the 2025 payload label.
std::string bTagHFPayloadSource(const std::string &year, std::string_view source) {
    if (!bTagHFSourceAvailable(year, source)) return {};
    if (year == "2025" && source == "topmass") return "mass";
    return std::string(source);
}

std::string bTagHFBranchName(std::string_view source) {
    return "weightsyst_btag_HF_" + std::string(source);
}

bool bTagHFSourceIsCoupled(std::string_view source) {
    // Generic JES/JER are coupled to the corresponding kinematic event
    // variations; 2025 regrouped JES sources are kept private and materialized
    // as companions of the matching analysis JES variations.
    return source == "pileup" || source == "isrdef" || source == "fsrdef" || source == "muf" || source == "mur" ||
           source == "jes" || source == "jer";
}

bool bTagHFSourceIsYearDecorrelated(std::string_view source) {
    return source == "uncorrelated" || source == "statistic";
}

std::string bTagHFInternalBranchName(std::string_view source) {
    return "_btagging_sf_HF_" + std::string(source);
}

void recordBTagFailure(const char *reason, std::string_view source, const char *direction,
                       const char *flavor, const char *category) {
    std::ostringstream key;
    key << reason << " source=" << source << " direction=" << direction
        << " flavor=" << flavor << " category=" << category;
    std::lock_guard<std::mutex> lock(g_btag_diagnostic_mutex);
    ++g_btag_failure_details[key.str()];
}

template <typename T>
RVec<T> correlateWeightWithBTagSource(const RVec<T> &raw, const RVec<double> &btag) {
    if (raw.size() != 3 || btag.size() != 3 || !std::isfinite(btag[0]) ||
        std::abs(btag[0]) < kBTagDenominatorEpsilon)
        throw std::runtime_error("Cannot correlate analysis weight with an invalid central HF b-tag factor");
    return RVec<T>{raw[0], static_cast<T>(raw[1] * btag[1] / btag[0]),
                   static_cast<T>(raw[2] * btag[2] / btag[0])};
}

RVec<double> bTagKinematicVariationRatios(const RVec<double> &btag) {
    return correlateWeightWithBTagSource<double>(RVec<double>{1., 1., 1.}, btag);
}

struct BTagWeightBundle {
    std::array<RVec<double>, kBTagHFSources.size()> hf;
    RVec<double> lf_uncorrelated = {1., 1., 1.};
    RVec<double> lf_correlated = {1., 1., 1.};
};

double unityForInvalidBTagWeight(std::atomic<unsigned long long> &counter);

std::string bTagObservedCategory(const std::vector<bool> &passed,
                                 const std::vector<std::size_t> &indices) {
    int passed_index = -1;
    for (int index = static_cast<int>(passed.size()) - 1; index >= 0; --index)
        if (passed[static_cast<std::size_t>(index)]) { passed_index = index; break; }
    if (passed_index < 0)
        return "fail" + std::string(kBTagInclusiveWorkingPoints[indices.front()]);
    if (passed_index == static_cast<int>(passed.size()) - 1)
        return std::string(kBTagInclusiveWorkingPoints[indices.back()]);
    return std::string(kBTagInclusiveWorkingPoints[indices[static_cast<std::size_t>(passed_index)]]) + "not" +
           std::string(kBTagInclusiveWorkingPoints[indices[static_cast<std::size_t>(passed_index + 1)]]);
}

int bTagTightestPassed(const std::vector<bool> &passed) {
    for (int index = static_cast<int>(passed.size()) - 1; index >= 0; --index)
        if (passed[static_cast<std::size_t>(index)]) return index;
    return -1;
}

double bTagCategoryWeight(const std::vector<double> &sf, const std::vector<double> &eff,
                          const std::vector<bool> &passed,
                          const std::vector<std::size_t> &indices,
                          std::string_view source, const char *direction,
                          const char *flavor) {
    if (sf.empty() || sf.size() != eff.size() || sf.size() != passed.size() || sf.size() != indices.size())
        throw std::runtime_error("Inconsistent b-tag working-point vector sizes");
    const std::string category = bTagObservedCategory(passed, indices);
    const auto fail = [&](const char *reason, std::atomic<unsigned long long> &counter) {
        recordBTagFailure(reason, source, direction, flavor, category.c_str());
        return unityForInvalidBTagWeight(counter);
    };
    for (std::size_t index = 0; index < passed.size(); ++index) {
        if (!std::isfinite(sf[index]) || !std::isfinite(eff[index]))
            return fail("invalid_probability", g_btag_invalid_probability);
        if (index > 0 && passed[index] && !passed[index - 1])
            return fail("nonnested_working_points", g_btag_invalid_probability);
        if (!(0. <= eff[index] && eff[index] <= 1.))
            return fail("invalid_probability", g_btag_invalid_probability);
    }
    for (std::size_t index = 1; index < eff.size(); ++index)
        if (eff[index - 1] < eff[index])
            return fail("invalid_probability", g_btag_invalid_probability);

    std::vector<double> q(sf.size());
    for (std::size_t index = 0; index < q.size(); ++index) {
        q[index] = sf[index] * eff[index];
        if (!std::isfinite(q[index]) || q[index] < 0. || q[index] > 1.)
            return fail("invalid_probability", g_btag_invalid_probability);
        if (index > 0 && q[index - 1] < q[index])
            return fail("invalid_probability", g_btag_invalid_probability);
    }

    const int passed_index = bTagTightestPassed(passed);
    if (passed_index < 0) {
        if (std::abs(1. - eff[0]) < kBTagDenominatorEpsilon)
            return fail("tiny_denominator", g_btag_tiny_denominator);
        return (1. - q[0]) / (1. - eff[0]);
    }
    if (passed_index == static_cast<int>(passed.size()) - 1) return sf.back();
    const std::size_t next = static_cast<std::size_t>(passed_index + 1);
    if (q[static_cast<std::size_t>(passed_index)] < q[next])
        return fail("negative_intermediate", g_btag_negative_intermediate);
    if (std::abs(eff[static_cast<std::size_t>(passed_index)] - eff[next]) < kBTagDenominatorEpsilon)
        return fail("tiny_denominator", g_btag_tiny_denominator);
    return (q[static_cast<std::size_t>(passed_index)] - q[next]) /
           (eff[static_cast<std::size_t>(passed_index)] - eff[next]);
}

std::size_t bTagHFSourceIndex(std::string_view source) {
    const auto it = std::find(kBTagHFSources.begin(), kBTagHFSources.end(), source);
    if (it == kBTagHFSources.end()) throw std::runtime_error("Unknown HF b-tag source " + std::string(source));
    return std::distance(kBTagHFSources.begin(), it);
}

constexpr const char *kBTagFamilyConfig = "corrections/scalefactors/btagging/btag_eff_families.yaml";

struct BTagFamilyConfig {
    std::vector<std::pair<std::string, std::vector<std::string>>> preliminary;
    std::map<std::string, std::vector<std::string>> final_samples;
    std::map<std::string, std::vector<std::string>> final_channels;
    std::vector<std::string> excluded_channels;
};

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t");
    return value.substr(begin, end - begin + 1);
}

BTagFamilyConfig loadBTagFamilyConfig(const std::string &path) {
    BTagFamilyConfig parsed;
    {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot read b-tag family configuration " + path);
        std::set<std::string> preliminary_names, excluded_names;
        std::string line, section, final_kind, current_group;
        const auto fail = [](const std::string &message) {
            throw std::runtime_error("Invalid canonical b-tag family YAML: " + message);
        };
        while (std::getline(input, line)) {
            if (line.find('\t') != std::string::npos) fail("tabs are not supported");
            const auto comment = line.find('#');
            if (comment != std::string::npos) line.erase(comment);
            const auto content = trim(line);
            if (content.empty()) continue;
            const auto indent = line.find_first_not_of(" \t");
            if (indent == 0) {
                if (content != "preliminary_families:" && content != "final_merges:" &&
                    content != "excluded_source_channels:") fail("unknown top-level key " + content);
                section = content.substr(0, content.size() - 1);
                final_kind.clear();
                current_group.clear();
                continue;
            }
            if (section == "preliminary_families") {
                if (indent == 2 && content.back() == ':') {
                    current_group = content.substr(0, content.size() - 1);
                    if (current_group.empty() || !preliminary_names.insert(current_group).second)
                        fail("duplicate or empty preliminary family");
                    parsed.preliminary.emplace_back(current_group, std::vector<std::string>{});
                } else if (indent == 4 && content.rfind("- ", 0) == 0 && !parsed.preliminary.empty()) {
                    parsed.preliminary.back().second.push_back(trim(content.substr(2)));
                } else fail("invalid preliminary_families indentation or syntax");
                continue;
            }
            if (section == "excluded_source_channels") {
                if (indent != 2 || content.rfind("- ", 0) != 0) fail("invalid excluded_source_channels entry");
                const auto channel = trim(content.substr(2));
                if (channel.empty() || !excluded_names.insert(channel).second) fail("duplicate or empty excluded channel");
                parsed.excluded_channels.push_back(channel);
                continue;
            }
            if (section != "final_merges") fail("content outside a supported YAML section");
            if (indent == 2 && content.back() == ':') {
                final_kind = content.substr(0, content.size() - 1);
                if (final_kind != "samples" && final_kind != "channels") fail("unknown final_merges kind " + final_kind);
                current_group.clear();
            } else if (indent == 4 && content.back() == ':' && !final_kind.empty()) {
                current_group = trim(content.substr(0, content.size() - 1));
                auto &groups = final_kind == "samples" ? parsed.final_samples : parsed.final_channels;
                if (current_group.empty() || !groups.emplace(current_group, std::vector<std::string>{}).second)
                    fail("duplicate or empty final group");
            } else if (indent == 6 && content.rfind("- ", 0) == 0 && !current_group.empty()) {
                auto &groups = final_kind == "samples" ? parsed.final_samples : parsed.final_channels;
                const auto member = trim(content.substr(2));
                if (member.empty()) fail("empty final group member");
                groups.at(current_group).push_back(member);
            } else fail("invalid final_merges indentation or syntax");
        }
        if (parsed.preliminary.empty() || parsed.final_samples.empty() || parsed.final_channels.empty() ||
            parsed.excluded_channels.empty()) fail("missing required non-empty mapping");
        for (const auto &[family, needles] : parsed.preliminary)
            if (needles.empty() || std::any_of(needles.begin(), needles.end(), [](const auto &x) { return x.empty(); }))
                fail("empty preliminary family " + family);
        const auto validate_groups = [&fail](const auto &groups, const std::set<std::string> &expected, const char *kind) {
            std::map<std::string, unsigned int> membership;
            for (const auto &[group, members] : groups) {
                if (members.empty()) fail(std::string("empty final ") + kind + " group " + group);
                for (const auto &member : members) ++membership[member];
            }
            std::set<std::string> observed;
            for (const auto &[member, count] : membership) {
                if (count != 1) fail(std::string("duplicate final ") + kind + " member " + member);
                observed.insert(member);
            }
            if (observed != expected) fail(std::string("final ") + kind + " membership does not match canonical sources");
        };
        validate_groups(parsed.final_samples, preliminary_names, "sample");
        std::set<std::string> retained_channels;
        for (const auto &[_, members] : parsed.final_channels)
            retained_channels.insert(members.begin(), members.end());
        validate_groups(parsed.final_channels, retained_channels, "channel");
        for (const auto &channel : parsed.excluded_channels)
            if (retained_channels.count(channel)) fail("excluded channel is also retained: " + channel);
    }
    return parsed;
}

const BTagFamilyConfig &bTagFamilyConfig(const std::string &year) {
    static const BTagFamilyConfig config = loadBTagFamilyConfig(kBTagFamilyConfig);
    if (year == "2016preVFP" || year == "2016postVFP" || year == "2017" ||
        year == "2018" || year == "2024Prompt") return config;
    throw std::runtime_error("No b-tag efficiency YAML is configured for unsupported year " + year);
}

std::string bTagEfficiencyPayloadYear(std::string_view year) {
    return year == "2025" ? "2024Prompt" : std::string(year);
}

double unityForInvalidBTagWeight(std::atomic<unsigned long long> &counter) {
    ++counter;
    return 1.;
}

std::string finalGroup(const std::map<std::string, std::vector<std::string>> &groups,
                       const std::string &name, const std::string &kind) {
    std::string match;
    for (const auto &[group, members] : groups) {
        if (std::find(members.begin(), members.end(), name) == members.end()) continue;
        if (!match.empty()) throw std::runtime_error(name + " occurs in multiple final b-tag " + kind + " groups");
        match = group;
    }
    if (match.empty()) throw std::runtime_error(name + " is not assigned to a final b-tag " + kind + " group");
    return match;
}

std::string bTagEfficiencyFamily(const std::string &sample, const std::string &year) {
    const auto &config = bTagFamilyConfig(year);
    for (const auto &[family, needles] : config.preliminary) {
        for (const auto &needle : needles) {
            if (sample.find(needle) != std::string::npos)
                return finalGroup(config.final_samples, family, "sample");
        }
    }
    throw std::runtime_error("No preliminary b-tag efficiency family is configured for sample " + sample);
}

std::string bTagEfficiencyChannel(const std::string &channel, const std::string &year) {
    const std::string canonical_channel =
        channel == "0lep_1FJ_met" ? "0lep_1FJ" :
        channel == "0lep_2FJ_met" ? "0lep_2FJ" : channel;
    if (canonical_channel == "all_events")
        throw std::runtime_error("all_events has no channel-specific b-tag efficiency payload. "
                                 "Specify an analysis channel, or rerun with --skip-btag-sf.");
    return finalGroup(bTagFamilyConfig(year).final_channels, canonical_channel, "channel");
}
} // namespace

void resetBTagDiagnostics() {
    g_btag_negative_intermediate = 0;
    g_btag_tiny_denominator = 0;
    g_btag_invalid_probability = 0;
    std::lock_guard<std::mutex> lock(g_btag_diagnostic_mutex);
    g_btag_failure_details.clear();
}

void printBTagDiagnostics(std::ostream &out) {
    const auto negative = g_btag_negative_intermediate.load();
    const auto tiny_denominator = g_btag_tiny_denominator.load();
    const auto invalid_probability = g_btag_invalid_probability.load();
    std::lock_guard<std::mutex> lock(g_btag_diagnostic_mutex);
    if (negative == 0 && tiny_denominator == 0 && invalid_probability == 0 &&
        g_btag_failure_details.empty()) return;
    out << "[BTag SF diagnostics] negative intermediate probabilities=" << negative
        << ", tiny denominators=" << tiny_denominator
        << ", invalid efficiencies/probabilities=" << invalid_probability << '\n';
    for (const auto &[detail, count] : g_btag_failure_details)
        out << "  " << count << " × " << detail << '\n';
}

correction::CorrectionSet loadBTagEfficiencyCorrectionSet(const std::string &year) {
    const std::string path = "corrections/scalefactors/btagging/btag_eff_" + year + ".json";
    return *CorrectionSet::from_file(path);
}

BTagEfficiencyContexts prepareBTagEfficiencyContexts(
    const std::vector<BTagEfficiencyMetadata> &metadata,
    const std::string &channel,
    const std::vector<std::string> &working_points) {
    for (const auto &wp : working_points) {
        if (std::find(kBTagInclusiveWorkingPoints.begin(), kBTagInclusiveWorkingPoints.end(), wp) ==
            kBTagInclusiveWorkingPoints.end())
            throw std::runtime_error("Unknown b-tag working point " + wp);
    }

    BTagEfficiencyContexts contexts;
    std::map<std::string, std::shared_ptr<correction::CorrectionSet>> efficiency_sets;
    for (const auto &entry : metadata) {
        const auto sf_set = bTaggingScaleFactors.find(entry.year);
        const auto hf_name = bTaggingScaleFactors_HF_corrname.find(entry.year);
        const auto lf_name = bTaggingScaleFactors_LF_corrname.find(entry.year);
        if (sf_set == bTaggingScaleFactors.end() || hf_name == bTaggingScaleFactors_HF_corrname.end() ||
            lf_name == bTaggingScaleFactors_LF_corrname.end())
            throw std::runtime_error("B-tag SF application is unsupported for year " + entry.year +
                                     " or its HF/LF payload names are not configured");

        const std::string efficiency_year = bTagEfficiencyPayloadYear(entry.year);
        const std::string efficiency_channel = bTagEfficiencyChannel(channel, efficiency_year);
        const std::string efficiency_sample = bTagEfficiencyFamily(entry.sample, efficiency_year);
        const std::string efficiency_name = "btag_" + efficiency_year + "_" + efficiency_channel;
        auto &efficiency_set = efficiency_sets[efficiency_year];
        if (!efficiency_set)
            efficiency_set = std::make_shared<correction::CorrectionSet>(loadBTagEfficiencyCorrectionSet(efficiency_year));

        BTagCorrectionRef efficiency;
        try { efficiency = efficiency_set->at(efficiency_name); }
        catch (...) {
            throw std::runtime_error("B-tag efficiency correction is unavailable for year=" + entry.year +
                                     ", efficiency_payload_year=" + efficiency_year +
                                     ", requested_correction=" + efficiency_name);
        }
        const auto hf_sf = sf_set->second.at(hf_name->second);
        const auto lf_sf = sf_set->second.at(lf_name->second);
        for (const auto &wp : working_points) {
            try {
                (void)hf_sf->evaluate({"central", wp, 5, 0., 30.});
                (void)lf_sf->evaluate({"central", wp, 0, 0., 30.});
                for (const auto source : {std::string("correlated"), std::string("uncorrelated")}) {
                    for (const auto direction : {std::string("up_"), std::string("down_")}) {
                        (void)lf_sf->evaluate({direction + source, wp, 0, 0., 30.});
                    }
                }
            } catch (const std::exception &error) {
                throw std::runtime_error("B-tag SF payload is missing year=" + entry.year +
                                         ", WP=" + wp + ": " + error.what());
            }
            const auto jes_scheme = bTagJESSchemeForYear(entry.year);
            const auto validate_hf_jes = [&](std::string_view analysis_source,
                                             std::string_view payload_source,
                                             const std::string &direction,
                                             int flavor) {
                try {
                    (void)hf_sf->evaluate({direction + std::string(payload_source), wp, flavor, 0., 30.});
                } catch (const std::exception &error) {
                    throw std::runtime_error("B-tag JES SF payload is missing year=" + entry.year +
                                             ", analysis_source=" + std::string(analysis_source) +
                                             ", payload_source=" + std::string(payload_source) +
                                             ", WP=" + wp + ", flavor=" + std::to_string(flavor) +
                                             ", direction=" + direction + ": " + error.what());
                }
            };
            if (jes_scheme == BTagJESScheme::Total) {
                for (const auto direction : {std::string("up_"), std::string("down_")})
                    for (const int flavor : {5, 4})
                        validate_hf_jes("jesTotal", "jes", direction, flavor);
            } else if (jes_scheme == BTagJESScheme::Regrouped11) {
                for (const auto &mapping : kBTagJES2025Mapping)
                    for (const auto direction : {std::string("up_"), std::string("down_")})
                        for (const int flavor : {5, 4})
                            validate_hf_jes(mapping.analysis_source, mapping.payload_source, direction, flavor);
            }
        }
        for (const auto flavor : {std::string("B"), std::string("C"), std::string("L")}) {
            for (const auto &wp : working_points) {
                try { (void)efficiency->evaluate({efficiency_sample, flavor, wp, 30., 0.}); }
                catch (...) {
                    throw std::runtime_error("B-tag efficiency payload is missing flavor=" + flavor +
                                             ", WP=" + wp + ", year=" + entry.year +
                                             ", efficiency_payload_year=" + efficiency_year +
                                             ", requested_channel=" + channel +
                                             ", final_channel=" + efficiency_channel +
                                             ", sample=" + entry.sample +
                                             ", final_sample_family=" + efficiency_sample +
                                             ", correction=" + efficiency_name);
                }
            }
        }
        contexts.emplace(std::make_pair(entry.year, entry.sample), BTagEfficiencyContext{
            efficiency_sample, efficiency_name, efficiency_set, efficiency, hf_sf, lf_sf});
    }
    if (contexts.empty()) throw std::runtime_error("No MC samples were found for b-tag SF validation");
    return contexts;
}

/*
############################################
GOLDEN JSON
############################################
*/

RNode applyGoldenJSONWeight(const lumiMask& golden, RNode df){
    auto goldenjson = [&golden](unsigned int &run, unsigned int &luminosityBlock){ return golden.accept(run, luminosityBlock); };
    return df.Filter(goldenjson, {"run", "luminosityBlock"});
}

/*
############################################
PILEUP SFs
############################################
*/

RNode applyPileupScaleFactors(std::unordered_map<std::string, correction::CorrectionSet> cset_pileup, std::unordered_map<std::string, std::string> year_map, RNode df) {
    auto eval_correction = [cset_pileup, year_map] (std::string year, float ntrueint) {
        RVec<double> pileup_weights;
        if (cset_pileup.find(year) == cset_pileup.end()) {
            std::string known;
            for (const auto& [k, _] : cset_pileup) known += (known.empty() ? "" : ", ") + k;
            throw std::runtime_error("Pileup: no correction set configured for year '" + year
                                     + "'. Configured years: " + known
                                     + ". Add the era to pileupScaleFactors and "
                                       "pileupScaleFactors_yearmap in src/weights.h.");
        }
        if (year_map.find(year) == year_map.end()) {
            throw std::runtime_error("Pileup: year '" + year + "' has a correction set but no entry in "
                                     "pileupScaleFactors_yearmap (src/weights.h).");
        }
        auto correctionset = cset_pileup.at(year).at(year_map.at(year));
        pileup_weights.push_back(correctionset->evaluate({ntrueint, "nominal"}));
        pileup_weights.push_back(correctionset->evaluate({ntrueint, "up"}));
        pileup_weights.push_back(correctionset->evaluate({ntrueint, "down"}));
        return pileup_weights;
    };
    return df.Define("_weight_pileup_raw", eval_correction, {"year", "Pileup_nTrueInt"});
}

/*
############################################
Lepton SFs (putting e and m together)
############################################
*/

// In weights.cpp - update the function:
RNode lepSFWrapper(RNode df,
    bool isData,
    const std::string& ele_sf_name,
    const std::string& muo_sf_name,
    bool include_trigger_sf,
    const std::string& ele_output_name,
    const std::string& muo_output_name)
{
    // Early return for data - no SFs needed
    if (isData) return df;

    std::string final_ele_sf = ele_sf_name;
    std::string final_muo_sf = muo_sf_name;

    // Optionally include trigger SFs
    if (include_trigger_sf) {
        // Apply trigger SFs if not already present
        auto colNames = df.GetColumnNames();
        bool has_ele_trigger = std::find(colNames.begin(), colNames.end(), "_weight_electrontrigger") != colNames.end();
        bool has_mu_trigger = std::find(colNames.begin(), colNames.end(), "_weight_muon_trigger") != colNames.end();

        if (!has_ele_trigger) {
            df = applyElectronTriggerScaleFactors(electronTriggerScaleFactors, electronTriggerScaleFactors_yearmap, df);
        }
        if (!has_mu_trigger) {
            df= applyMuonScaleFactors(muonScaleFactors, "_weight_muon_trigger", muonSFConfigs.at("_weight_muon_trigger"), df);
        }

        // Multiply base SFs by trigger SFs
        auto multiply_sf = [](const RVec<double>& base_sf, const RVec<double>& trig_sf) {
            return RVec<double>{ base_sf[0]*trig_sf[0], base_sf[1]*trig_sf[1], base_sf[2]*trig_sf[2] };
        };
        df = df.Define("_" + ele_output_name + "_ele_with_trigger", multiply_sf, {ele_sf_name, "_weight_electrontrigger"});
        df = df.Define("_" + muo_output_name + "_muo_with_trigger", multiply_sf, {muo_sf_name, "_weight_muon_trigger"});

        final_ele_sf = "_" + ele_output_name + "_ele_with_trigger";
        final_muo_sf = "_" + muo_output_name + "_muo_with_trigger";
    }

    // Create the separate electron and muon SF outputs with custom names
    df = df.Define(ele_output_name, final_ele_sf);
    df = df.Define(muo_output_name, final_muo_sf);

    // Optionally update main weight
    auto update_weight = [](double current_weight, const RVec<double>& ele_sf, const RVec<double>& muo_sf) {
        return current_weight * ele_sf[0] * muo_sf[0];
    };
    df = df.Redefine("weight", update_weight, {"weight", final_ele_sf, final_muo_sf});

    return df;
}

/*
############################################
MUON SFs
############################################
*/

RNode applyMuonScaleFactors(
    std::unordered_map<std::string, MuonCorrectionSet> cset_muon,
    std::string output_name,
    MuonSFConfig config,
    RNode df)
{
    auto eval_correction = [cset_muon, config] (std::string year, const RVec<float> eta, const RVec<float> pt) {
        RVec<double> muon_sf_weights = {1., 1., 1.};
        if (eta.empty()) {
            return muon_sf_weights;
        }
        if (cset_muon.find(year) == cset_muon.end()) {
            static std::unordered_set<std::string> warned_years;
            if (warned_years.find(year) == warned_years.end()) {
                std::cout << "Warning: Muon SF correction set for year " << year << " not found. Setting weights to 1." << std::endl;
                warned_years.insert(year);
            }
            return muon_sf_weights;
        }
        if (config.year_map.find(year) == config.year_map.end()) {
            return muon_sf_weights;
        }

        const auto& entry = cset_muon.at(year);
        const auto& year_config = config.year_map.at(year);
        auto correctionset = entry.cset.at(year_config.correction_key);

        for (size_t i = 0; i < eta.size(); i++) {
            float eta_to_pass = entry.abs_eta ? abs(eta[i]) : eta[i];
            float pt_to_pass = std::max(pt[i], year_config.pt_min);

            muon_sf_weights[0] *= correctionset->evaluate({eta_to_pass, pt_to_pass, "nominal"});
            muon_sf_weights[1] *= correctionset->evaluate({eta_to_pass, pt_to_pass, "systup"});
            muon_sf_weights[2] *= correctionset->evaluate({eta_to_pass, pt_to_pass, "systdown"});
        }
        return muon_sf_weights;
    };

    return df.Define(output_name, eval_correction, {"year", "_muonSel_eta", "_muonSel_pt"});
}

RNode combineScaleFactorWeightsByKey(RNode df, std::string output_name, std::vector<std::string> input_keys)
{
    auto combine = [](const RVec<double>& w0,
                      const RVec<double>& w1,
                      const RVec<double>& w2) {
        return RVec<double>{
            w0[0] * w1[0] * w2[0],
            w0[1] * w1[1] * w2[1],
            w0[2] * w1[2] * w2[2]
        };
    };

    return df.Define(output_name, combine, input_keys);
}


RNode applyMuonWorkingPointSFs(RNode df, bool isData, std::vector<std::string> wp_keys)
{
    if (isData) return df;

    std::unordered_set<std::string> component_keys;

    for (const auto& wp_key : wp_keys) {
        if (muonWorkingPointSFs.find(wp_key) == muonWorkingPointSFs.end()) {
            throw std::runtime_error("applyMuonWorkingPointSFs: unknown WP key '" + wp_key + "'");
        }
        for (const auto& component_key : muonWorkingPointSFs.at(wp_key)) {
            component_keys.insert(component_key);
        }
    }

    for (const auto& component_key : component_keys) {
        if (muonSFConfigs.find(component_key) == muonSFConfigs.end()) {
            throw std::runtime_error("applyMuonWorkingPointSFs: unknown component key '" + component_key + "'");
        }
        df = applyMuonScaleFactors(muonScaleFactors, component_key, muonSFConfigs.at(component_key), df);
    }

    for (const auto& wp_key : wp_keys) {
        df = combineScaleFactorWeightsByKey(df, wp_key, muonWorkingPointSFs.at(wp_key));
    }

    return df;
}


/*
############################################
ELECTRON SFs
############################################
*/

// The "year" axis inside the EGM jsons is EGM's campaign label, which does not always
// match our era label -- we call the 2025 era "2025", EGM calls it "2025Prompt". Every
// other era happens to coincide, so map only where it differs.
static std::string egmYearKey(const std::string& year) {
    static const std::unordered_map<std::string, std::string> m = {
        {"2025", "2025Prompt"},
    };
    auto it = m.find(year);
    return (it == m.end()) ? year : it->second;
}


RNode applyElectronRecoScaleFactors(std::unordered_map<std::string, correction::CorrectionSet> cset_electron, RNode df, std::string output_name) {
    auto eval_correction = [cset_electron] (std::string year, const RVec<float> eta, const RVec<float> pt) {
        RVec<double> electron_sf_weights = {1., 1., 1.};
        if (eta.empty()) {
            return electron_sf_weights;
        }
        if (cset_electron.find(year) == cset_electron.end()) {
            static std::unordered_set<std::string> warned_years;
            if (warned_years.find(year) == warned_years.end()) {
                std::cout << "Warning: Electron Reco correction set for year " << year << " not found. Setting electron reco weights to 1." << std::endl;
                warned_years.insert(year);
            }
            return electron_sf_weights;
        }

        bool is_run2 = (year.find("2016") != std::string::npos ||
                        year.find("2017") != std::string::npos ||
                        year.find("2018") != std::string::npos);

        std::string correction_name = is_run2 ? "UL-Electron-ID-SF" : "Electron-ID-SF";
        auto correctionset = cset_electron.at(year).at(correction_name);
        const std::string egm_year = egmYearKey(year);

        for (size_t i = 0; i < eta.size(); i++) {
            if (is_run2) {
                if (pt[i] >= 20) {
                    electron_sf_weights[0] *= correctionset->evaluate({egm_year, "sf",     "RecoAbove20", eta[i], pt[i]});
                    electron_sf_weights[1] *= correctionset->evaluate({egm_year, "sfup",   "RecoAbove20", eta[i], pt[i]});
                    electron_sf_weights[2] *= correctionset->evaluate({egm_year, "sfdown", "RecoAbove20", eta[i], pt[i]});
                } else {
                    const auto& low_pt = electronRecoLowPt.at(year);
                    float pt_to_pass = std::max(pt[i], low_pt.pt_min);
                    electron_sf_weights[0] *= correctionset->evaluate({egm_year, "sf",     low_pt.working_point, eta[i], pt_to_pass});
                    electron_sf_weights[1] *= correctionset->evaluate({egm_year, "sfup",   low_pt.working_point, eta[i], pt_to_pass});
                    electron_sf_weights[2] *= correctionset->evaluate({egm_year, "sfdown", low_pt.working_point, eta[i], pt_to_pass});
                }
            } else {
                if (pt[i] >= 20 && pt[i] < 75) {
                    electron_sf_weights[0] *= correctionset->evaluate({egm_year, "sf",     "Reco20to75", eta[i], pt[i]});
                    electron_sf_weights[1] *= correctionset->evaluate({egm_year, "sfup",   "Reco20to75", eta[i], pt[i]});
                    electron_sf_weights[2] *= correctionset->evaluate({egm_year, "sfdown", "Reco20to75", eta[i], pt[i]});
                } else if (pt[i] >= 75) {
                    electron_sf_weights[0] *= correctionset->evaluate({egm_year, "sf",     "RecoAbove75", eta[i], pt[i]});
                    electron_sf_weights[1] *= correctionset->evaluate({egm_year, "sfup",   "RecoAbove75", eta[i], pt[i]});
                    electron_sf_weights[2] *= correctionset->evaluate({egm_year, "sfdown", "RecoAbove75", eta[i], pt[i]});
                } else {
                    const auto& low_pt = electronRecoLowPt.at(year);
                    float pt_to_pass = std::max(pt[i], low_pt.pt_min);
                    electron_sf_weights[0] *= correctionset->evaluate({egm_year, "sf",     low_pt.working_point, eta[i], pt_to_pass});
                    electron_sf_weights[1] *= correctionset->evaluate({egm_year, "sfup",   low_pt.working_point, eta[i], pt_to_pass});
                    electron_sf_weights[2] *= correctionset->evaluate({egm_year, "sfdown", low_pt.working_point, eta[i], pt_to_pass});
                }
            }
        }
        return electron_sf_weights;
    };
    return df.Define(output_name, eval_correction, {"year", "_electronSel_SC_eta", "_electronSel_pt"});
}

RNode applyElectronIDScaleFactors(std::unordered_map<std::string, correction::CorrectionSet> cset_electron, ElectronIDConfig config, std::string output_name, RNode df) {
    auto eval_correction = [cset_electron, config] (std::string year, const RVec<float> eta, const RVec<float> pt) {
        RVec<double> electron_sf_weights = {1., 1., 1.};
        if (eta.empty()) {
            return electron_sf_weights;
        }
        if (cset_electron.find(year) == cset_electron.end()) {
            static std::unordered_set<std::string> warned_years;
            if (warned_years.find(year) == warned_years.end()) {
                std::cout << "Warning: Electron ID correction set for year " << year << " not found. Setting electron ID weights to 1." << std::endl;
                warned_years.insert(year);
            }
            return electron_sf_weights;
        }
        if (config.correction_name_map.find(year) == config.correction_name_map.end()) {
            return electron_sf_weights;
        }

        auto correctionset = cset_electron.at(year).at(config.correction_name_map.at(year));
        const std::string egm_year = egmYearKey(year);
        for (size_t i = 0; i < eta.size(); i++) {
            electron_sf_weights[0] *= correctionset->evaluate({egm_year, "sf",     config.working_point, eta[i], pt[i]});
            electron_sf_weights[1] *= correctionset->evaluate({egm_year, "sfup",   config.working_point, eta[i], pt[i]});
            electron_sf_weights[2] *= correctionset->evaluate({egm_year, "sfdown", config.working_point, eta[i], pt[i]});
        }
        return electron_sf_weights;
    };
    return df.Define(output_name, eval_correction, {"year", "_electronSel_SC_eta", "_electronSel_pt"});
}

RNode combineElectronScaleFactorWeightsByKey(RNode df, std::string output_name, std::vector<std::string> input_keys)
{
    auto combine = [](const RVec<double>& w0,
                      const RVec<double>& w1) {
        return RVec<double>{
            w0[0] * w1[0],
            w0[1] * w1[1],
            w0[2] * w1[2]
        };
    };

    return df.Define(output_name, combine, input_keys);
}

RNode applyElectronWorkingPointSFs(RNode df, bool isData, std::vector<std::string> wp_keys)
{
    if (isData) return df;

    bool need_reco = false;
    bool need_id_loose = false;
    bool need_id_tight = false;

    for (const auto& wp_key : wp_keys) {
        if (wp_key == "_weight_electron_reco_looseid") {
            need_reco = true;
            need_id_loose = true;
        }
        else if (wp_key == "_weight_electron_reco_tightid") {
            need_reco = true;
            need_id_tight = true;
        }
        else {
            throw std::runtime_error("applyElectronWorkingPointSFs: unknown WP key '" + wp_key + "'");
        }
    }

    if (need_reco) {
        df = applyElectronRecoScaleFactors(electronScaleFactors, df, "_weight_electron_reco");
    }
    if (need_id_loose) {
        df = applyElectronIDScaleFactors(electronScaleFactors, electronID_loose, "_weight_electron_id_loose", df);
    }
    if (need_id_tight) {
        df = applyElectronIDScaleFactors(electronScaleFactors, electronID_tight, "_weight_electron_id_tight", df);
    }

    for (const auto& wp_key : wp_keys) {
        df = combineElectronScaleFactorWeightsByKey(df, wp_key, electronWorkingPointSFs.at(wp_key));
    }

    return df;
}

/*
############################################
ELECTRON TRIGGER SFs
############################################
*/

RNode applyElectronTriggerScaleFactors(std::unordered_map<std::string, correction::CorrectionSet> cset_electron, std::unordered_map<std::string, std::string> year_map, RNode df) {
    auto eval_correction = [cset_electron, year_map] (std::string year, const RVec<float> eta, const RVec<float> pt) {
        RVec<double> electron_sf_weights = {1., 1., 1.};
        if (eta.empty()) {
            return electron_sf_weights;
        }
        if (cset_electron.find(year) == cset_electron.end()) {
            static std::unordered_set<std::string> warned_years;
            if (warned_years.find(year) == warned_years.end()) {
                std::cout << "Warning: Electron Trigger correction set for year " << year << " not found. Setting electron trigger weights to 1." << std::endl;
                warned_years.insert(year);
            }
            return electron_sf_weights;
        }
        auto correctionset = cset_electron.at(year).at(year_map.at(year));
        const std::string egm_year = egmYearKey(year);
        for (size_t i = 0; i < eta.size(); i++) {
            electron_sf_weights[0] *= correctionset->evaluate({egm_year, "sf",     "HLT_SF_Ele30_TightID", eta[i], pt[i]});
            electron_sf_weights[1] *= correctionset->evaluate({egm_year, "sfup",   "HLT_SF_Ele30_TightID", eta[i], pt[i]});
            electron_sf_weights[2] *= correctionset->evaluate({egm_year, "sfdown", "HLT_SF_Ele30_TightID", eta[i], pt[i]});
        }
        return electron_sf_weights;
    };
    return df.Define("_weight_electrontrigger", eval_correction, {"year", "electron_SC_eta", "electron_pt"});
}

/*
############################################
BTAG SFs
############################################
*/


RNode applyBTaggingScaleFactors(const std::string &channel,
                                 const std::vector<std::string> &working_points,
                                 const BTagEfficiencyContexts &contexts, RNode df) {
    std::vector<std::size_t> selected_indices;
    std::vector<std::string> selected_wp_names;
    for (const auto &wp : working_points) {
        const auto it = std::find(kBTagInclusiveWorkingPoints.begin(), kBTagInclusiveWorkingPoints.end(), wp);
        if (it == kBTagInclusiveWorkingPoints.end()) throw std::runtime_error("Unknown b-tag working point " + wp);
        selected_indices.push_back(static_cast<std::size_t>(std::distance(kBTagInclusiveWorkingPoints.begin(), it)));
        selected_wp_names.push_back(wp);
    }
    const bool evaluate_regrouped_btag_jes = !jesVariationSuffixes().empty();
    auto evaluate_bundle = [contexts, selected_indices, selected_wp_names, evaluate_regrouped_btag_jes]
        (const std::string &year, const std::string &sample,
         const RVec<float> &eta,
         const RVec<float> &pt, const RVec<unsigned char> &jetflavor,
         const RVec<bool> &is_loose, const RVec<bool> &is_medium,
         const RVec<bool> &is_tight, const RVec<bool> &is_extra_tight,
         const RVec<bool> &is_extra_extra_tight) {
        if (eta.size() != pt.size() || eta.size() != jetflavor.size() ||
            eta.size() != is_loose.size() || eta.size() != is_medium.size() ||
            eta.size() != is_tight.size() || eta.size() != is_extra_tight.size() ||
            eta.size() != is_extra_extra_tight.size())
            throw std::runtime_error("B-tag input collections have inconsistent sizes");
        const auto context = contexts.find({year, sample});
        if (context == contexts.end())
            throw std::runtime_error("No validated b-tag context for year=" + year + ", sample=" + sample);
        const auto &ctx = context->second;
        BTagWeightBundle bundle;
        for (auto &weights : bundle.hf) weights = {1., 1., 1.};
        for (std::size_t jet = 0; jet < pt.size(); ++jet) {
            const int flavor = std::abs(jetflavor[jet]);
            if (std::abs(eta[jet]) >= bTagMaxAbsEta(year)) continue;
            const bool heavy = flavor == 5 || flavor == 4;
            const char *label = flavor == 5 ? "B" : (flavor == 4 ? "C" : "L");
            const std::array<bool, kBTagInclusiveWorkingPoints.size()> all_passed = {
                is_loose[jet], is_medium[jet], is_tight[jet],
                is_extra_tight[jet], is_extra_extra_tight[jet]
            };
            const int sf_flavor = heavy ? flavor : 0;
            const auto &sf = heavy ? ctx.hf_sf : ctx.lf_sf;
            const double sf_eta = bTagSFAbsEta(year, eta[jet]);
            const auto evaluate_sf = [&](const std::string &systematic, int sf_flavor_value) {
                std::vector<double> values;
                values.reserve(selected_indices.size());
                for (const auto &wp : selected_wp_names)
                    values.push_back(sf->evaluate({systematic,
                                                   wp,
                                                   sf_flavor_value, sf_eta, pt[jet]}));
                return values;
            };
            std::vector<bool> passed;
            std::vector<double> selected_efficiencies, selected_central_sf;
            passed.reserve(selected_indices.size());
            selected_efficiencies.reserve(selected_indices.size());
            selected_central_sf.reserve(selected_indices.size());
            for (std::size_t selected = 0; selected < selected_indices.size(); ++selected) {
                const auto index = selected_indices[selected];
                passed.push_back(all_passed[index]);
                    selected_efficiencies.push_back(ctx.efficiency->evaluate({ctx.efficiency_sample, label,
                                                                       selected_wp_names[selected],
                                                                       pt[jet], eta[jet]}));
            }
            selected_central_sf = evaluate_sf("central", sf_flavor);
            // The selected cumulative efficiencies retain the full exclusive
            // content: for non-adjacent WPs, e.g. [L,T], eps_L-eps_T is the
            // sum of the omitted LnotM and MnotT intervals.
            const double central_weight = bTagCategoryWeight(selected_central_sf, selected_efficiencies, passed, selected_indices,
                                                              "central", "central", label);
            if (!heavy) {
                for (const auto source : {std::string("uncorrelated"), std::string("correlated")}) {
                    auto &weights = source == "uncorrelated" ? bundle.lf_uncorrelated : bundle.lf_correlated;
                    weights[0] *= central_weight;
                    for (const auto direction : {std::string("up_"), std::string("down_")}) {
                        const auto selected_shifted_sf = evaluate_sf(direction + source, sf_flavor);
                        const double shifted_weight = bTagCategoryWeight(selected_shifted_sf, selected_efficiencies, passed, selected_indices,
                                                                          source, direction == "up_" ? "up" : "down", label);
                        weights[direction == "up_" ? 1 : 2] *= shifted_weight;
                    }
                }
                continue;
            }
            for (std::size_t index = 0; index < kBTagHFSources.size(); ++index) {
                const std::string source(kBTagHFSources[index]);
                auto &weights = bundle.hf[index];
                weights[0] *= central_weight;
                const std::string payload_source = bTagHFPayloadSource(year, source);
                if (payload_source.empty() ||
                    (bTagHFSourceIsRegroupedJES(source) && !evaluate_regrouped_btag_jes)) {
                    weights[1] *= central_weight;
                    weights[2] *= central_weight;
                    continue;
                }
                for (const auto direction : {std::string("up_"), std::string("down_")}) {
                    std::vector<double> selected_shifted_sf;
                    selected_shifted_sf.reserve(selected_indices.size());
                    for (std::size_t selected = 0; selected < selected_indices.size(); ++selected) {
                        try {
                            const double payload = ctx.hf_sf->evaluate({direction + payload_source,
                                                                     selected_wp_names[selected],
                                                                     flavor, sf_eta, pt[jet]});
                            const double central = selected_central_sf[selected];
                            selected_shifted_sf.push_back(flavor == 4 ? central + 2. * (payload - central) : payload);
                        } catch (const std::exception &error) {
                            throw std::runtime_error("B-tag SF evaluation failed for year=" + year + ", source=" + source +
                                                     ", payload_source=" + payload_source + ", direction=" + direction +
                                                     ", flavor=" + std::to_string(flavor) +
                                                     ", wp=" + selected_wp_names[selected] +
                                                     ": " + error.what());
                        }
                    }
                    const double shifted_weight = bTagCategoryWeight(
                        selected_shifted_sf, selected_efficiencies, passed, selected_indices, source,
                        direction == "up_" ? "up" : "down", label);
                    weights[direction == "up_" ? 1 : 2] *= shifted_weight;
                }
            }
        }
        return bundle;
    };

    auto result = df.Define("_btagging_sf_bundle", evaluate_bundle,
                            {"year", "name", "jet_eta", "jet_pt", "jet_hadronFlavour",
                             "jet_isLooseBTag", "jet_isMediumBTag", "jet_isTightBTag",
                             "jet_isExtraTightBTag", "jet_isExtraExtraTightBTag"});
    const std::set<std::string> context_years = [&contexts] {
        std::set<std::string> years;
        for (const auto &[key, _] : contexts) years.insert(key.first);
        return years;
    }();
    for (const auto source : kBTagHFSources) {
        const auto index = bTagHFSourceIndex(source);
        bool present_in_input = false;
        for (const auto &year : context_years)
            present_in_input = present_in_input || !bTagHFPayloadSource(year, source).empty();
        if (!present_in_input) continue;
        if (bTagHFSourceIsRegroupedJES(source)) {
            result = result.Define(bTagHFInternalBranchName(source),
                                   [index](const BTagWeightBundle &bundle) { return bundle.hf[index]; }, {"_btagging_sf_bundle"});
        } else if (bTagHFSourceIsCoupled(source)) {
            result = result.Define(bTagHFInternalBranchName(source),
                                   [index](const BTagWeightBundle &bundle) { return bundle.hf[index]; }, {"_btagging_sf_bundle"});
        } else if (bTagHFSourceIsYearDecorrelated(source)) {
            for (const auto &year : context_years) {
                if (bTagHFPayloadSource(year, source).empty()) continue;
                const std::string branch = bTagHFBranchName(source) + "_" + bTagSafeYearToken(year);
                result = result.Define(branch,
                    [index, year](const std::string &event_year, const BTagWeightBundle &bundle) {
                        const auto &weights = bundle.hf[index];
                        return event_year == year ? weights : RVec<double>{weights[0], weights[0], weights[0]};
                    }, {"year", "_btagging_sf_bundle"});
            }
        } else {
            result = result.Define(bTagHFBranchName(source),
                                   [index](const BTagWeightBundle &bundle) { return bundle.hf[index]; }, {"_btagging_sf_bundle"});
        }
    }
    for (const auto &year : context_years) {
        const std::string branch = "weightsyst_btag_LF_uncorrelated_" + bTagSafeYearToken(year);
        result = result.Define(branch,
            [year](const std::string &event_year, const BTagWeightBundle &bundle) {
                return event_year == year ? bundle.lf_uncorrelated : RVec<double>{bundle.lf_uncorrelated[0], bundle.lf_uncorrelated[0], bundle.lf_uncorrelated[0]};
            }, {"year", "_btagging_sf_bundle"});
    }
    return result.Define("weightsyst_btag_LF_correlated", [](const BTagWeightBundle &bundle) { return bundle.lf_correlated; }, {"_btagging_sf_bundle"});
}

bool bTagHFSourcePresentInContexts(const BTagEfficiencyContexts &contexts, std::string_view source) {
    for (const auto &[key, _] : contexts)
        if (!bTagHFPayloadSource(key.first, source).empty()) return true;
    return false;
}

/*
############################################
OTHER SFs
############################################
*/

// See https://github.com/cmstas/run3-vbsvvh/pull/28#issuecomment-3820814039
RNode applyEWKCorrections(correction::CorrectionSet cset_ewk, RNode df){
    auto eval_correction = [cset_ewk] (RVec<float> LHEPart_pt, RVec<float> LHEPart_eta, RVec<float> LHEPart_phi, RVec<float> LHEPart_mass, RVec<int> LHEPart_pdgId, int do_ewk_corr) {
        if(do_ewk_corr == 0) return 1.;
        else{
            TLorentzVector TEWKq1, TEWKq2, TEWKlep, TEWKnu;
            TEWKq1.SetPtEtaPhiM(LHEPart_pt[4],LHEPart_eta[4],LHEPart_phi[4],LHEPart_mass[4]);
            TEWKq2.SetPtEtaPhiM(LHEPart_pt[5],LHEPart_eta[5],LHEPart_phi[5],LHEPart_mass[5]);
            TEWKlep.SetPtEtaPhiM(LHEPart_pt[2],LHEPart_eta[2],LHEPart_phi[2],LHEPart_mass[2]);
            TEWKnu.SetPtEtaPhiM(LHEPart_pt[3],LHEPart_eta[3],LHEPart_phi[3],LHEPart_mass[3]);
            int chargequark[7] = {0,-1,2,-1,2,-1,2};
            int EWKpdgq1 = LHEPart_pdgId[4];
            int EWKpdgq2 = LHEPart_pdgId[5];
            int EWKsignq1 = (EWKpdgq1 > 0) - (EWKpdgq1 < 0);
            int EWKsignq2 = (EWKpdgq2 > 0) - (EWKpdgq2 < 0);
            double EWKMass_q12 = (TEWKq1 + TEWKq2).M();
            double EWKMass_lnu = (TEWKlep + TEWKnu).M();
            double fabscharge=(fabs((double)(EWKsignq1 * chargequark[abs(EWKpdgq1)] + (EWKsignq2 * chargequark[abs(EWKpdgq2)]))))/3;
            double EWKbjet_pt = -999;
            if(fabscharge ==1){
                if( EWKMass_q12 >= 70 && EWKMass_q12 < 90  && 
                    EWKMass_lnu >= 70 && EWKMass_lnu < 90){
                    return 0.;
                }
            }
            if(EWKMass_q12 >= 95){
                if( abs(EWKpdgq1) == 5 && abs(EWKpdgq2) == 5){
                    if(TEWKq1.Pt() > TEWKq2.Pt())  EWKbjet_pt = TEWKq1.Pt();
                    else                           EWKbjet_pt = TEWKq2.Pt();
                }else if(abs(EWKpdgq1) == 5){
                    EWKbjet_pt = TEWKq1.Pt();
                }else if(abs(EWKpdgq2) == 5){
                    EWKbjet_pt = TEWKq2.Pt();
                }
            }
            if(EWKbjet_pt > -998){
                return cset_ewk.at("EWK")->evaluate({EWKbjet_pt});
            }
            else return 1.;
        }
    };
    return df.Define("ewkweight", eval_correction, {"LHEPart_pt", "LHEPart_eta", "LHEPart_phi", "LHEPart_mass", "LHEPart_pdgId", "do_ewk_corr"});
}

RNode applyL1PreFiringReweighting(RNode df){
    // L1PreFiringWeight_* branches are only present in Run 2 NanoAOD; the
    // correction does not apply to Run 3, so emit a unit weight there.
    auto colNames = df.GetColumnNames();
    bool hasL1Prefire = std::find(colNames.begin(), colNames.end(), std::string("L1PreFiringWeight_Nom")) != colNames.end();
    if (!hasL1Prefire) {
        return df.Define("weightsyst_l1prefiring", [] () { return RVec<float>{1.f, 1.f, 1.f}; }, {});
    }
    auto eval_correction = [] (float L1prefire, float L1prefireup, float L1prefiredown) {
        return RVec<float>{L1prefire, L1prefireup, L1prefiredown};
    };
    return df.Define("weightsyst_l1prefiring", eval_correction, {"L1PreFiringWeight_Nom", "L1PreFiringWeight_Up", "L1PreFiringWeight_Dn"});
}

RNode applyPSWeight_FSR(RNode df) {
    auto eval_correction = [] (const RVec<float> PSWeight) {
        return RVec<float>{1., PSWeight[1], PSWeight[3]};
    };
    return df.Define("_weight_PSFSR_raw", eval_correction, {"PSWeight"});
}

RNode applyPSWeight_ISR(RNode df) {
    auto eval_correction = [] (const RVec<float> PSWeight) {
        return RVec<float>{1., PSWeight[0], PSWeight[2]};
    };
    return df.Define("_weight_PSISR_raw", eval_correction, {"PSWeight"});
}

RNode applyLHEScaleWeight_muF(RNode df) {
    auto eval_correction = [] (const RVec<float> LHEScaleWeight) {
        return RVec<float>{1., LHEScaleWeight[5], LHEScaleWeight[3]};
    };
    return df.Define("_weight_muF_raw", eval_correction, {"LHEScaleWeight"});
}

RNode applyLHEScaleWeight_muR(RNode df) {
    auto eval_correction = [] (const RVec<float> LHEScaleWeight) {
        return RVec<float>{1., LHEScaleWeight[7], LHEScaleWeight[1]};
    };
    return df.Define("_weight_muR_raw", eval_correction, {"LHEScaleWeight"});
}

RNode applyDataWeights(RNode df_) {
    return applyGoldenJSONWeight(LumiMask, df_);
}

RNode applyMCWeights(RNode df_, const std::string &channel, bool apply_btag_sf,
                     const std::vector<std::string> &btag_working_points,
                     const BTagEfficiencyContexts &btag_contexts) {
    // Check for LHE branches (not present in all samples, e.g. QCD)
    auto colNames = df_.GetColumnNames();
    auto hasColumn = [&colNames](const std::string& name) {
        return std::find(colNames.begin(), colNames.end(), name) != colNames.end();
    };
    bool hasLHEScale = hasColumn("LHEScaleWeight");
    bool hasLHEPart = hasColumn("LHEPart_pt");

    auto df = applyPileupScaleFactors(pileupScaleFactors, pileupScaleFactors_yearmap, df_);

    if (apply_btag_sf) {
        resetBTagDiagnostics();
        df = applyBTaggingScaleFactors(channel, btag_working_points, btag_contexts, df);
    }

    if (hasLHEPart) {
        df = applyEWKCorrections(cset_ewk, df);
    } else {
        df = df.Define("ewkweight", [] () { return 1.; }, {});
    }

    df = applyL1PreFiringReweighting(df);
    df = applyPSWeight_FSR(df);
    df = applyPSWeight_ISR(df);

    if (hasLHEScale) {
        df = applyLHEScaleWeight_muF(df);
        df = applyLHEScaleWeight_muR(df);
    } else {
        df = df.Define("_weight_muF_raw", [] () { return RVec<float>{1.f, 1.f, 1.f}; }, {});
        df = df.Define("_weight_muR_raw", [] () { return RVec<float>{1.f, 1.f, 1.f}; }, {});
    }

    // Preserve the original analysis variations untouched.  The optional
    // *_withbSF branches carry the corresponding b-tag response separately.
    df = df.Define("weight_pileup", [](const RVec<double> &weight) { return weight; }, {"_weight_pileup_raw"})
           .Define("weight_PSISR", [](const RVec<float> &weight) { return weight; }, {"_weight_PSISR_raw"})
           .Define("weight_PSFSR", [](const RVec<float> &weight) { return weight; }, {"_weight_PSFSR_raw"})
           .Define("weight_muF", [](const RVec<float> &weight) { return weight; }, {"_weight_muF_raw"})
           .Define("weight_muR", [](const RVec<float> &weight) { return weight; }, {"_weight_muR_raw"});
    if (apply_btag_sf) {
        if (bTagHFSourcePresentInContexts(btag_contexts, "pileup"))
            df = df.Define("weightsyst_pileup_withbSF", correlateWeightWithBTagSource<double>,
                           {"_weight_pileup_raw", bTagHFInternalBranchName("pileup")});
        if (bTagHFSourcePresentInContexts(btag_contexts, "isrdef"))
            df = df.Define("weightsyst_PSISR_withbSF", correlateWeightWithBTagSource<float>,
                           {"_weight_PSISR_raw", bTagHFInternalBranchName("isrdef")});
        if (bTagHFSourcePresentInContexts(btag_contexts, "fsrdef"))
            df = df.Define("weightsyst_PSFSR_withbSF", correlateWeightWithBTagSource<float>,
                           {"_weight_PSFSR_raw", bTagHFInternalBranchName("fsrdef")});
        if (bTagHFSourcePresentInContexts(btag_contexts, "muf"))
            df = df.Define("weightsyst_muF_withbSF", correlateWeightWithBTagSource<float>,
                           {"_weight_muF_raw", bTagHFInternalBranchName("muf")});
        if (bTagHFSourcePresentInContexts(btag_contexts, "mur"))
            df = df.Define("weightsyst_muR_withbSF", correlateWeightWithBTagSource<float>,
                           {"_weight_muR_raw", bTagHFInternalBranchName("mur")});
    }

    // Keep `weight_*` for the untouched event-level analysis variations.
    // B-tag-correlated variations use the `weightsyst_*` namespace.

    df = df.Define("weightsyst_pileup", [](const RVec<double> &weight) { return weight; }, {"weight_pileup"})
           .Define("weightsyst_PSISR", [](const RVec<float> &weight) { return weight; }, {"weight_PSISR"})
           .Define("weightsyst_PSFSR", [](const RVec<float> &weight) { return weight; }, {"weight_PSFSR"})
           .Define("weightsyst_muF", [](const RVec<float> &weight) { return weight; }, {"weight_muF"})
           .Define("weightsyst_muR", [](const RVec<float> &weight) { return weight; }, {"weight_muR"});

    // The generic JES/JER source responses are event-level variation
    // factors. Define them only when that payload source exists in the input.
    if (apply_btag_sf) {
        if (bTagHFSourcePresentInContexts(btag_contexts, "jes"))
            df = df.Define("weightsyst_jes", bTagKinematicVariationRatios,
                           {bTagHFInternalBranchName("jes")});
        if (bTagHFSourcePresentInContexts(btag_contexts, "jer"))
            df = df.Define("weightsyst_jer", bTagKinematicVariationRatios,
                           {bTagHFInternalBranchName("jer")});

        // 2025 BTV JES responses are companions of the matching analysis JES
        // variations, not independent nuisances. Only materialize them when
        // the corresponding analysis JES columns are being stored.
        if (!jesVariationSuffixes().empty()) {
            for (const auto &mapping : kBTagJES2025Mapping) {
                const std::string payload_source(mapping.payload_source);
                if (!bTagHFSourcePresentInContexts(btag_contexts, payload_source)) continue;
                df = df.Define("weightsyst_jes" + std::string(mapping.analysis_source),
                               bTagKinematicVariationRatios,
                               {bTagHFInternalBranchName(payload_source)});
            }
        }
    }

    std::string nominal_weight = std::string("weight *") +
        "weightsyst_pileup[0] * "
        // Lepton SF wrappers update the nominal `weight` directly and expose
        // their variation vectors separately, so do not multiply them twice.
        "";
    if (apply_btag_sf)
        nominal_weight += "weightsyst_btag_HF_correlated[0] * "
                          "weightsyst_btag_LF_correlated[0] * ";
    nominal_weight +=
        "ewkweight * "
        "weightsyst_l1prefiring[0] * "
        "weightsyst_PSISR[0] * "
        "weightsyst_PSFSR[0] * "
        "weightsyst_muF[0] * "
        "weightsyst_muR[0]";
    return df.Redefine("weight", nominal_weight);
}
