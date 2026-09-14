#include "corrections.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "TRandom3.h"

/*
############################################
CORRECTION-SET REGISTRY — single definition point for all correctionlib payloads
############################################

Everything in this section used to be namespace-scope map definitions in
corrections.h. Namespace-scope `const` has internal linkage, so every
translation unit including the header built its own private copy — each
payload was gzip-decompressed and parsed once per TU, at static-init time
before main() (where a missing cvmfs file meant std::terminate with no usable
message). The maps are now function-local statics: defined once, loaded
lazily on first use after main() starts (thread-safe per C++11 magic statics).

The 11 formerly hand-written JERC maps (paths, JEC prefixes/suffixes, year
tokens, JER res/sf names for AK4 and AK8) are derived from the single era
table below — adding or re-pinning an era means editing ONE table row.
*/

namespace {

using CSetMap = std::unordered_map<std::string, correction::CorrectionSet>;
using StrMap  = std::unordered_map<std::string, std::string>;

/*
JERC ERA TABLE
--------------
Run 2 + the 2022/2023 Run 3 eras are pinned to the 2026-06-05 JME snapshot
(jet_jerc / fatJet_jerc); the 2024Prompt and 2025 eras are pinned to the newer
2026-07-16 snapshot (JEC V5/V3, JER JRV2). JER-Smearing is pinned to
2025-11-03 — NOT the mutable `latest/` symlink. The tag strings below are the
compound/correction key roots that exist inside those pinned files. The
"2024Prompt" era uses the Summer24Prompt24 tag (JEC V5 / JER JRV2), and the
"2025" era (2025 data + Summer24 MC) uses the JME-recommended Summer24Prompt25
tag (JEC V3 / JER JRV2). The 2026-07-14 update moved the L2L3Residual
corrections to signed-η (from |η|) and refreshed the JER SFs (evalJECCompound
already passes signed eta and resolves inputs by name, so no code change was
needed); 2026-07-16 fixed the V4/V2 DATA L2L3Residual payloads that 2026-07-14
had failed to actually update (→ tags V5/V3).

correctionlib has no "give me the newest version" API: the code asks for a tag
by name, so the file snapshot and the requested tag string must be bumped
together, deliberately, in one table row. Loading from `latest/` while pinning
the tag string is what previously let the two drift apart (V3 code vs V4 file
-> silent no-op; now a hard error, see applyJetEnergyCorrections below).

To adopt a newer JME release: pick the new dated dir, find the new recommended
tag, and update the snapshot + tags in the row.

Derived name conventions (see the accessors below):
  JEC compound     : <jecTag>_{MC,DATA}_L1L2L3Res_<algo>
  JES uncertainty  : <jecTag>_MC_<source>[_<yearToken>]_<algo>
  JER res / sf     : <jerTag>_MC_{PtResolution,ScaleFactor}_<algo>
with <algo> = AK4PFPuppi (jet_jerc) or AK8PFPuppi (fatJet_jerc). The year
token appears in year-decorrelated Regrouped source names (e.g.
"Summer20UL18NanoV15_V1_MC_Regrouped_Absolute_2018_AK4PFPuppi").
*/
struct EraJERC {
    std::string jmeDir;     // era directory under /cvmfs/cms-griddata.cern.ch/cat/metadata/JME
    std::string snapshot;   // pinned dated snapshot subdir — NOT the mutable `latest/`
    std::string jecTag;     // JEC release, without the trailing _MC/_DATA
    std::string jerTag;     // JER release, without _MC_{PtResolution,ScaleFactor}_<algo>
    std::string yearToken;  // token in year-decorrelated Regrouped JES source names
};

const std::map<std::string, EraJERC>& eraJERCTable() {
    static const std::map<std::string, EraJERC> table = {
        //  era                      JME era directory                                        snapshot      JEC tag                       JER tag                            year token
        {"2016preVFP",           {"Run2-2016preVFP-UL-NanoAODv15",                        "2026-06-05", "Summer20UL16APVNanoV15_V1",  "Summer20UL16APV_JRV5",            "2016APV"}},
        {"2016postVFP",          {"Run2-2016postVFP-UL-NanoAODv15",                       "2026-06-05", "Summer20UL16NanoV15_V1",     "Summer20UL16_JRV5",               "2016"}},
        {"2017",                 {"Run2-2017-UL-NanoAODv15",                              "2026-06-05", "Summer20UL17NanoV15_V1",     "Summer19UL17_JRV4",               "2017"}},
        {"2018",                 {"Run2-2018-UL-NanoAODv15",                              "2026-06-05", "Summer20UL18NanoV15_V1",     "Summer19UL18_JRV3",               "2018"}},
        {"2022Re-recoBCD",       {"Run3-22CDSep23-Summer22-NanoAODv12",                   "2026-06-05", "Summer22_22Sep2023_V4",      "Summer22_22Sep2023_JRV2",         "2022"}},
        {"2022Re-recoE+PromptFG",{"Run3-22EFGSep23-Summer22EE-NanoAODv12",                "2026-06-05", "Summer22EE_22Sep2023_V4",    "Summer22EE_22Sep2023_JRV2",       "2022EE"}},
        {"2023PromptC",          {"Run3-23CSep23-Summer23-NanoAODv12",                    "2026-06-05", "Summer23Prompt23_V4",        "Summer23Prompt23_RunCv1234_JRV2", "2023"}},
        {"2023PromptD",          {"Run3-23DSep23-Summer23BPix-NanoAODv12",                "2026-06-05", "Summer23BPixPrompt23_V4",    "Summer23BPixPrompt23_RunD_JRV2",  "2023BPix"}},
        // 2024 + 2025: pinned to 2026-07-16, which fixes a 2026-07-14 bug where the V4/V2 DATA
        // L2L3Residual payloads had not actually been updated (JEC tags bumped to V5/V3 with the fix).
        {"2024Prompt",           {"Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2026-07-16", "Summer24Prompt24_V5",        "Summer24Prompt24_JRV2",           "2024"}},
        // 2025 data + Summer24 MC — JME-recommended Summer24Prompt25 (JEC + JER + JES self-contained).
        {"2025",                 {"Run3-25Prompt-Summer24-NanoAODv15",                    "2026-07-16", "Summer24Prompt25_V3",        "Summer24Prompt25_JRV2",           "2025"}},
    };
    return table;
}

const std::string kJMEBase = "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/";

CSetMap loadEraJERCSets(const std::string& filename) {
    CSetMap out;
    for (const auto& [year, era] : eraJERCTable())
        out.emplace(year, *CorrectionSet::from_file(
            kJMEBase + era.jmeDir + "/" + era.snapshot + "/" + filename));
    return out;
}

template <typename MakeValue>
StrMap perEra(MakeValue&& make) {
    StrMap out;
    for (const auto& [year, era] : eraJERCTable()) out.emplace(year, make(era));
    return out;
}

// --- JERC payloads + derived name maps (all keyed by the era table above) ------------

// AK4 and AK8 JEC/JER payloads — jet_jerc / fatJet_jerc under the same pinned era dirs.
const CSetMap& jetEnergyCorrections()    { static const CSetMap m = loadEraJERCSets("jet_jerc.json.gz");    return m; }
const CSetMap& fatJetEnergyCorrections() { static const CSetMap m = loadEraJERCSets("fatJet_jerc.json.gz"); return m; }

// JER hybrid-smearing formula (era-independent), pinned to 2025-11-03.
const CSetMap& jetEnergyResolution_smear() {
    static const CSetMap m = [] {
        CSetMap out;
        out.emplace("jer_smear",
                    *CorrectionSet::from_file(kJMEBase + "JER-Smearing/2025-11-03/jer_smear.json.gz"));
        return out;
    }();
    return m;
}

// `<jecTag>_MC` — compound/uncertainty name prefix. Identical for AK4 and AK8 (both live
// under the same JEC release), so there is a single prefix map. makeCompoundJECName()
// swaps the trailing _MC for _DATA when running on data.
const StrMap& jetEnergyCorrections_JEC_prefix() {
    static const StrMap m = perEra([](const EraJERC& e) { return e.jecTag + "_MC"; });
    return m;
}

// Algo suffixes. The value is a property of the jet collection, not the era, but the
// JEC/JES helpers look their inputs up per year, so these stay year-keyed maps.
const StrMap& jetEnergyCorrections_JEC_suffix() {
    static const StrMap m = perEra([](const EraJERC&) { return std::string("AK4PFPuppi"); });
    return m;
}
const StrMap& fatJetEnergyCorrections_JEC_suffix() {
    static const StrMap m = perEra([](const EraJERC&) { return std::string("AK8PFPuppi"); });
    return m;
}

// Year token embedded in the year-decorrelated Regrouped JES source names.
const StrMap& jetEnergyCorrections_yearToken() {
    static const StrMap m = perEra([](const EraJERC& e) { return e.yearToken; });
    return m;
}

// JER PtResolution / ScaleFactor correction names: `<jerTag>_MC_{PtResolution,ScaleFactor}_<algo>`.
StrMap makeJERNames(const std::string& kind, const std::string& algo) {
    return perEra([&](const EraJERC& e) { return e.jerTag + "_MC_" + kind + "_" + algo; });
}
const StrMap& jetEnergyResolution_JER_res_name()    { static const StrMap m = makeJERNames("PtResolution", "AK4PFPuppi"); return m; }
const StrMap& jetEnergyResolution_JER_sf_name()     { static const StrMap m = makeJERNames("ScaleFactor",  "AK4PFPuppi"); return m; }
const StrMap& fatJetEnergyResolution_JER_res_name() { static const StrMap m = makeJERNames("PtResolution", "AK8PFPuppi"); return m; }
const StrMap& fatJetEnergyResolution_JER_sf_name()  { static const StrMap m = makeJERNames("ScaleFactor",  "AK8PFPuppi"); return m; }

// JER SF uncertainty (split-tag format, JME 2026-06-04 "newJERFormats" update). The pinned
// ScaleFactor corrections carry NO `systematic` input; the ±1σ SF is ScaleFactor ± SFUncertainty
// (verified against the last old-format payload: reproduces the old up/down to <=1e-4).
const StrMap& jetEnergyResolution_JER_unc_name()    { static const StrMap m = makeJERNames("SFUncertainty", "AK4PFPuppi"); return m; }
const StrMap& fatJetEnergyResolution_JER_unc_name() { static const StrMap m = makeJERNames("SFUncertainty", "AK8PFPuppi"); return m; }

// --- B-tagging -----------------------------------------------------------------------

// Note: Re-using the 2024 payload for 2022-2023 (same Run3-24... file).
// 2025 recommendation: https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Prompt25/
const CSetMap& btaggingCorrections() {
    static const CSetMap m = [] {
        const std::string base = "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/";
        const std::vector<std::pair<std::string, std::string>> eras = {
            {"2016preVFP",            "Run2-2016preVFP-UL-NanoAODv15"},
            {"2016postVFP",           "Run2-2016postVFP-UL-NanoAODv15"},
            {"2017",                  "Run2-2017-UL-NanoAODv15"},
            {"2018",                  "Run2-2018-UL-NanoAODv15"},
            {"2022Re-recoBCD",        "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15"},
            {"2022Re-recoE+PromptFG", "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15"},
            {"2023PromptC",           "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15"},
            {"2023PromptD",           "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15"},
            {"2024Prompt",            "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15"},
            {"2025",                  "Run3-25Prompt-Summer24-NanoAODv15"},
        };
        CSetMap out;
        for (const auto& [year, dir] : eras)
            out.emplace(year, *CorrectionSet::from_file(base + dir + "/latest/btagging.json.gz"));
        return out;
    }();
    return m;
}

// Numeric WP thresholds, one evaluate() per (era, WP) on first use.
std::unordered_map<std::string, float> makeBtagWPMap(const std::string& wp) {
    std::unordered_map<std::string, float> out;
    for (const auto& [year, cs] : btaggingCorrections())
        out.emplace(year, cs.at("UParTAK4_wp_values")->evaluate({wp}));
    return out;
}
const std::unordered_map<std::string, float>& btaggingWPMap_Loose()  { static const auto m = makeBtagWPMap("L"); return m; }
const std::unordered_map<std::string, float>& btaggingWPMap_Medium() { static const auto m = makeBtagWPMap("M"); return m; }
const std::unordered_map<std::string, float>& btaggingWPMap_Tight()  { static const auto m = makeBtagWPMap("T"); return m; }
const std::unordered_map<std::string, float>& btaggingWPMap_ExtraTight() {
    static const auto m = makeBtagWPMap("XT");
    return m;
}
const std::unordered_map<std::string, float>& btaggingWPMap_ExtraExtraTight() {
    static const auto m = makeBtagWPMap("XXT");
    return m;
}

// --- MET φ corrections ------------------------------------------------------------------

// FIXME: met corrections missing for v15
const CSetMap& metCorrections() {
    static const CSetMap m = {
        {"2016preVFP", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016preVFP-UL-NanoAODv9/latest/met.json.gz")},
        {"2016postVFP", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016postVFP-UL-NanoAODv9/latest/met.json.gz")},
        {"2017", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2017-UL-NanoAODv9/latest/met.json.gz")},
        {"2018", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2018-UL-NanoAODv9/latest/met.json.gz")},
        // {"2022Re-recoBCD", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/latest/met_xyCorrections_2022_2022.json.gz")},
        // {"2022Re-recoE+PromptFG", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/latest/met_xyCorrections_2022_2022EE.json.gz")},
        // {"2023PromptC", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/latest/met_xyCorrections_2023_2023.json.gz")},
        // {"2023PromptD", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/latest/met_xyCorrections_2023_2023BPix.json.gz")}
    };
    return m;
}

// --- Jet veto maps ------------------------------------------------------------------
/*
JET VETO MAP ERA TABLE
----------------------
Same pinning discipline as the JERC table above: a dated snapshot dir, NOT the
mutable `latest/` symlink, paired with the tag string that exists inside it.
correctionlib resolves the tag by name, so file and tag must be bumped together
in one row; loading `latest/` while hard-coding the tag is what lets the two
drift apart (a V2 re-publish landing under `latest/` would throw from
`CorrectionSet::at` inside the event loop).

Each snapshot below is the newest dated dir whose jetvetomaps.json.gz is
byte-identical to that era's `latest/` as of 2026-07-27, so this pinning is a
no-op on the payloads — it only removes the exposure to future re-publishes.
The dirs are the JME veto-map era dirs, which for Run 2 are the NanoAODv9 ones
(the NanoAODv15 dirs ship byte-identical maps; the veto maps are pure (eta,phi)
lookups and carry no NanoAOD-version dependence).

The map type we evaluate is 'jetvetomap' — the type the JSONs themselves
document as "the recommended map for analyses", for both Run 2 and Run 3.
Application differs by run period, per the JME prescription: Run 3 vetoes the whole EVENT if any jet lands in a veto region, Run 2
vetoes only the JETS themselves.
Run 2 recommendations: https://cms-jerc.web.cern.ch/Recommendations/#run-2_1
*/
struct EraVetoMap {
    std::string jmeDir;    // era directory under /cvmfs/cms-griddata.cern.ch/cat/metadata/JME
    std::string snapshot;  // pinned dated snapshot subdir — NOT the mutable `latest/`
    std::string tag;       // correction name inside that file
};

const std::map<std::string, EraVetoMap>& eraVetoMapTable() {
    static const std::map<std::string, EraVetoMap> table = {
        //  era                      JME era directory                                        snapshot      tag
        {"2016preVFP",           {"Run2-2016preVFP-UL-NanoAODv9",                         "2026-06-10", "Summer19UL16_V1"}},
        {"2016postVFP",          {"Run2-2016postVFP-UL-NanoAODv9",                        "2026-06-10", "Summer19UL16_V1"}},
        {"2017",                 {"Run2-2017-UL-NanoAODv9",                               "2026-06-10", "Summer19UL17_V1"}},
        {"2018",                 {"Run2-2018-UL-NanoAODv9",                               "2026-06-10", "Summer19UL18_V1"}},
        {"2022Re-recoBCD",       {"Run3-22CDSep23-Summer22-NanoAODv12",                   "2026-06-05", "Summer22_23Sep2023_RunCD_V1"}},
        {"2022Re-recoE+PromptFG",{"Run3-22EFGSep23-Summer22EE-NanoAODv12",                "2026-06-05", "Summer22EE_23Sep2023_RunEFG_V1"}},
        {"2023PromptC",          {"Run3-23CSep23-Summer23-NanoAODv12",                    "2026-07-15", "Summer23Prompt23_RunC_V1"}},
        {"2023PromptD",          {"Run3-23DSep23-Summer23BPix-NanoAODv12",                "2026-07-15", "Summer23BPixPrompt23_RunD_V1"}},
        {"2024Prompt",           {"Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15", "2026-07-16", "Summer24Prompt24_RunBCDEFGHI_V1"}},
        {"2025",                 {"Run3-25Prompt-Summer24-NanoAODv15",                    "2026-07-16", "Summer24Prompt25_RunCDEFG_V1"}},
    };
    return table;
}

const CSetMap& jetVetoMaps() {
    static const CSetMap m = [] {
        CSetMap out;
        for (const auto& [year, era] : eraVetoMapTable())
            out.emplace(year, *CorrectionSet::from_file(
                kJMEBase + era.jmeDir + "/" + era.snapshot + "/jetvetomaps.json.gz"));
        return out;
    }();
    return m;
}

const StrMap& jetVetoMap_names() {
    static const StrMap m = [] {
        StrMap out;
        for (const auto& [year, era] : eraVetoMapTable()) out.emplace(year, era.tag);
        return out;
    }();
    return m;
}

// --- Electron scale & smearing ---------------------------------------------------------
const CSetMap& electronSSCorrections() {
    static const CSetMap m = {
        {"2016preVFP",            *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2016preVFP-UL-NanoAODv15/2025-12-05/electronSS_EtDependent.json.gz")},
        {"2016postVFP",           *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2016postVFP-UL-NanoAODv15/2025-12-05/electronSS_EtDependent.json.gz")},
        {"2017",                  *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2017-UL-NanoAODv15/2025-12-05/electronSS_EtDependent.json.gz")},
        {"2018",                  *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2018-UL-NanoAODv15/2025-12-05/electronSS_EtDependent.json.gz")},
        {"2022Re-recoBCD",        *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22CDSep23-Summer22-NanoAODv12/2025-12-15/electronSS_EtDependent.json.gz")},
        {"2022Re-recoE+PromptFG", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-12-15/electronSS_EtDependent.json.gz")},
        {"2023PromptC",           *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23CSep23-Summer23-NanoAODv12/2025-12-15/electronSS_EtDependent.json.gz")},
        {"2023PromptD",           *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-12-15/electronSS_EtDependent.json.gz")},
        {"2024Prompt",            *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-15/electronSS_EtDependent.json.gz")},
        {"2025",                  *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-25Prompt-Summer24-NanoAODv15/2026-06-26/electronSS_EtDependent.json.gz")}
    };
    return m;
}

// --- JMS/JMR placeholder maps (unwired; template for the future GloParT calibration) ----
// See the JMS/JMR section in corrections.h. Identity-valued:
// JMS shift 0.0 GeV, JMR factor 1.0. σ_rel feeds the unmatched stochastic branch of
// applyJetMassResolution and is unreachable while the factor is 1.0 — replace both
// together with the per-era values from the calibration fit.
[[maybe_unused]] const std::unordered_map<std::string, float> jetMassScale_central = {
    {"2016preVFP", 0.0f}, {"2016postVFP", 0.0f}, {"2017", 0.0f}, {"2018", 0.0f},
    {"2022Re-recoBCD", 0.0f}, {"2022Re-recoE+PromptFG", 0.0f},
    {"2023PromptC", 0.0f}, {"2023PromptD", 0.0f},
    {"2024Prompt", 0.0f}
};

[[maybe_unused]] const std::unordered_map<std::string, float> jetMassResolution_central = {
    {"2016preVFP", 1.0f}, {"2016postVFP", 1.0f}, {"2017", 1.0f}, {"2018", 1.0f},
    {"2022Re-recoBCD", 1.0f}, {"2022Re-recoE+PromptFG", 1.0f},
    {"2023PromptC", 1.0f}, {"2023PromptD", 1.0f},
    {"2024Prompt", 1.0f}
};

[[maybe_unused]] const std::unordered_map<std::string, float> jetMassResolution_sigmaRel_central = {
    {"2016preVFP", 1.0f}, {"2016postVFP", 1.0f}, {"2017", 1.0f}, {"2018", 1.0f},
    {"2022Re-recoBCD", 1.0f}, {"2022Re-recoE+PromptFG", 1.0f},
    {"2023PromptC", 1.0f}, {"2023PromptD", 1.0f},
    {"2024Prompt", 1.0f}
};

} // anonymous namespace

/*
############################################
B-TAGGING WORKING POINTS
############################################
*/

RVec<bool> isbTagLoose(std::string year, RVec<float> btag_score) {
    return btag_score > btaggingWPMap_Loose().at(year);
}

RVec<bool> isbTagMedium(std::string year, RVec<float> btag_score) {
    return btag_score > btaggingWPMap_Medium().at(year);
}

RVec<bool> isbTagTight(std::string year, RVec<float> btag_score) {
    return btag_score > btaggingWPMap_Tight().at(year);
}

RVec<bool> isbTagExtraTight(std::string year, RVec<float> btag_score) {
    return btag_score > btaggingWPMap_ExtraTight().at(year);
}

RVec<bool> isbTagExtraExtraTight(std::string year, RVec<float> btag_score) {
    return btag_score > btaggingWPMap_ExtraExtraTight().at(year);
}

/*
############################################
JERC EVALUATION HELPERS — shared by the JEC / JER / Type-1 MET code below
############################################

makeCompoundJECName + evalJECCompound serve the nominal AK4/AK8 JEC and the
Type-1 MET rebuild; evalJERCorrection serves the AK4/AK8 JER smearing and the
CorrT1METJet JER inside applyType1MET.
*/

namespace {
    inline std::string makeCompoundJECName(const std::string& mc_prefix,
                                           const std::string& algo,
                                           bool isData) {
        // jec_prefix_map values end in "_MC" by convention; replace with "_DATA" for data.
        std::string base = mc_prefix;
        const std::string mc_tag = "_MC";
        // check base ends with "_MC" and chop those last 3 characters off
        if (base.size() >= mc_tag.size() &&
            base.compare(base.size() - mc_tag.size(), mc_tag.size(), mc_tag) == 0) {
            base.resize(base.size() - mc_tag.size());
        }
        return base + (isData ? "_DATA" : "_MC") + "_L1L2L3Res_" + algo;
    }

    // Evaluate a JEC L1L2L3Res compound, building the argument vector by matching each of the
    // correction's declared inputs() by name. This is deliberate: the compound's input list is
    // NOT fixed — it varies from era to era, differs between MC and DATA, and can change on a file
    // re-pin. Hard-coding a fixed-length arg list would silently break the moment a new file adds
    // or drops an input. Passing arguments by name stays correct across all of those without
    // touching this function; only a genuinely new input name reaches the `throw` below, which is
    // a loud, intentional signal to add a mapping (not a silent wrong answer).
    //
    // Example of the kind of drift this guards against: the 2026-06-05 files added a `JetPhi` input
    // to the 2023BPix + 2024 compounds (DATA there also takes `run`), while older eras have neither.
    // A fixed arg list threw "wrong number of inputs" for those eras, which the per-jet catch turned
    // into sf=1.0 — jets silently left at raw pT.
    inline double evalJECCompound(const correction::CompoundCorrection& comp,
                                  double area, double eta, double pt, double rho,
                                  double phi, double run) {
        std::vector<correction::Variable::Type> args;
        args.reserve(comp.inputs().size());
        for (const auto& v : comp.inputs()) {
            const std::string n = v.name();
            if      (n == "JetA")   args.push_back(area);
            else if (n == "JetEta") args.push_back(eta);
            else if (n == "JetPt")  args.push_back(pt);
            else if (n == "Rho")    args.push_back(rho);
            else if (n == "JetPhi") args.push_back(phi);
            else if (n == "run")    args.push_back(run);
            else throw std::runtime_error("JEC compound: unexpected input '" + n
                                          + "'. corrections.cpp evalJECCompound() needs a mapping for it.");
        }
        return comp.evaluate(args);
    }

    // Evaluate a JER PtResolution or ScaleFactor correction, building the argument vector by
    // matching each declared input() by name (same rationale as evalJECCompound above): the input
    // list is NOT fixed — it differs between the two correction types and can change on a file
    // re-pin — so position-based argument passing is fragile. Matching by name means one helper
    // serves both objects and survives file bumps; `syst` is consumed only when the correction
    // actually declares a `systematic` input (PtResolution does not; the ScaleFactor may or may not).
    //
    // Example of the kind of drift this guards against: the 2026-06-05 files ship the ScaleFactor
    // as [JetEta, JetPt] with NO `systematic`, whereas older files had one. A position-based
    // heuristic that assumed "3+ inputs ⇒ [JetEta, systematic, ...]" fed "nom" into the JetPt slot
    // ("wrong type: got string").
    inline double evalJERCorrection(const correction::Correction& c,
                                    double eta, double pt, double rho, const std::string& syst) {
        std::vector<correction::Variable::Type> args;
        args.reserve(c.inputs().size());
        for (const auto& v : c.inputs()) {
            const std::string n = v.name();
            if      (n == "JetEta")     args.push_back(eta);
            else if (n == "JetPt")      args.push_back(pt);
            else if (n == "Rho")        args.push_back(rho);
            else if (n == "systematic") args.push_back(syst);
            else throw std::runtime_error("JER correction '" + c.name() + "': unexpected input '" + n
                                          + "'. corrections.cpp evalJERCorrection() needs a mapping for it.");
        }
        return c.evaluate(args);
    }

    // One jet's JER smear factors {nom, up, dn} via JERSmear. The ±1σ scale factors are
    // sf ± unc — the pinned files use the split-tag format (JME 2026-06-04 "newJERFormats"):
    // the ScaleFactor correction has NO `systematic` axis and the symmetric absolute
    // uncertainty ships as a separate SFUncertainty correction. res, gen_pt, rho and the
    // event seed are shared by all three evaluations, so the stochastic Gaussian kick is
    // identical and the variations stay fully correlated with the nominal smearing, per the
    // JME hybrid recipe. With withVariations=false the up/dn slots repeat the nominal.
    // Two conventions applied HERE, on top of the raw JERSmear payload:
    //  1. The 3σ gen-match window (|pt − gen| < 3·res·pt) — the payload does NOT implement
    //     it (it only branches on the sign of GenPt); the caller must apply it, exactly as
    //     the JERC tutorial does (JecApplication.cpp::jerFactor). Out-of-window matches are
    //     demoted to the stochastic branch (gen := −1).
    //  2. Stochastic-branch seeding: in the payload, for GenPt<0 the JetPt input is used
    //     ONLY as hashprng seed material (the smear formula itself reads res/sf/N). We pass
    //     the variation-independent seed_pt (the RAW jet pt) there instead of the corrected
    //     pt, so the Gaussian draw is IDENTICAL for the nominal and every JES/JER variation
    //     — otherwise a per-mille pt shift would re-roll the random number and inject O(res)
    //     noise into the variation templates. (coffea's CorrectedJetsFactory seeds its
    //     Gaussian from the raw pt for the same reason.) Gen-matched jets keep the true pt
    //     (their branch is analytic in JetPt; no randomness).
    inline std::array<double, 3> evalJERSmearNomUpDn(const correction::Correction& smear,
                                                     double pt, double eta, double gen_pt,
                                                     double seed_pt,
                                                     double rho, unsigned long long event,
                                                     double res, double sf, double unc,
                                                     bool withVariations) {
        const bool useGen  = (gen_pt >= 0.0) && (std::abs(pt - gen_pt) < 3.0 * res * pt);
        const double ptIn  = useGen ? pt : seed_pt;
        const double genIn = useGen ? gen_pt : -1.0;
        // The payload formulas are NOT clamped: a very negative Gaussian (or a bad match)
        // can give a negative factor → negative jet pt. Clamp at 0 ("jet fluctuated away"),
        // per the JME documentation ("always ≥ 0"; coffea clamps to a minimal jet pt).
        auto ev = [&](double sfv) {
            return std::max(0.0, smear.evaluate({ptIn, eta, genIn, rho, (int)event, res, sfv}));
        };
        const double nom = ev(sf);
        if (!withVariations) return {nom, nom, nom};
        return {nom, ev(sf + unc), ev(sf - unc)};
    }

    // Per-collection JER smear factors for the nominal and the ±1σ SF variations
    // (RDF column payload; consumed by the Define chains below and by applyType1MET).
    struct JERFactors {
        RVec<float> nom, up, dn;
    };

    // Resolve the per-jet matched gen pt from a NanoAOD gen index column (−1 if unmatched).
    inline RVec<float> matchedGenPt(const RVec<short>& idx, const RVec<float>& gpt) {
        RVec<float> out(idx.size(), -1.0f);
        for (size_t i = 0; i < idx.size(); ++i)
            if (idx[i] >= 0 && idx[i] < (int)gpt.size()) out[i] = gpt[idx[i]];
        return out;
    }
}

/*
############################################
JET ENERGY CORRECTIONS — nominal
############################################

Recipe (per AK4 jet, year-aware):

  pt_raw   = (1 - Jet_rawFactor) * Jet_pt     // undo NanoAOD JEC
  mass_raw = (1 - Jet_rawFactor) * Jet_mass
  factor   = compound(Jet_area, Jet_eta, pt_raw, rho [, Jet_phi] [, run])
             // compound = L1FastJet * L2Relative * L3Absolute (* L2L3Residual for DATA),
             // evaluated internally by correctionlib with `inputs_update=[JetPt]` so each
             //  stage feeds the next its updated pt. The 2023BPix + 2024 pinned files also
             // require Jet_phi; evalJECCompound() appends JetPhi/run only when the file
             // declares them (see above).
  pt_new   = pt_raw   * factor
  mass_new = mass_raw * factor

MET is NOT touched here anymore. Type-I MET is rebuilt from RawPuppiMET in applyType1MET()
(over Jet + CorrT1METJet, with muon subtraction), following the new JERC recipe at
https://cms-jerc.web.cern.ch/Type1MET/. See the TYPE-1 PUPPI MET section below.

NanoAOD branches consumed:
  Jet_pt, Jet_mass, Jet_eta, Jet_phi, Jet_area, Jet_rawFactor,
  Rho_fixedGridRhoFastjetAll, run, year
*/

RNode applyJetEnergyCorrections(const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
                                const std::unordered_map<std::string, std::string>& jec_prefix_map,
                                const std::unordered_map<std::string, std::string>& jec_suffix_map,
                                RNode df, bool isData) {

    auto compute_factor = [cset_jerc, jec_prefix_map, jec_suffix_map, isData](
            std::string year,
            RVec<float> pt, RVec<float> eta, RVec<float> phi, RVec<float> area, RVec<float> rawFactor,
            float rho, unsigned int run) {

        RVec<float> factor(pt.size(), 1.0f);
        if (pt.empty()) return factor;

        auto cs_it = cset_jerc.find(year);
        auto pf_it = jec_prefix_map.find(year);
        auto sf_it = jec_suffix_map.find(year);
        if (cs_it == cset_jerc.end() || pf_it == jec_prefix_map.end() || sf_it == jec_suffix_map.end()) {
            throw std::runtime_error("AK4 JEC: no correction set, prefix, or suffix configured for year '"
                                     + year + "'. Refusing to run with jets uncorrected — add this year to "
                                       "the JERC era table in corrections.cpp.");
        }

        const std::string compound_name = makeCompoundJECName(pf_it->second, sf_it->second, isData);
        std::shared_ptr<const correction::CompoundCorrection> compound;
        try {
            compound = cs_it->second.compound().at(compound_name);
        } catch (const std::out_of_range&) {
            throw std::runtime_error("AK4 JEC: compound '" + compound_name
                                     + "' not found in the pinned jet_jerc.json.gz for year '" + year
                                     + "'. The JEC tag in the JERC era table (corrections.cpp) is stale relative to the "
                                       "pinned snapshot (version drift). Refusing to run with jets uncorrected.");
        }

        for (size_t i = 0; i < pt.size(); ++i) {
            float pt_raw = (1.0f - rawFactor[i]) * pt[i];
            double sf = 1.0;
            try {
                // evalJECCompound appends JetPhi / run only when the pinned file declares them.
                sf = evalJECCompound(*compound, (double)area[i], (double)eta[i], (double)pt_raw,
                                     (double)rho, (double)phi[i], (double)run);
            } catch (const std::exception& e) {
                std::cout << "Warning: JEC evaluation failed for jet " << i << " (pt_raw=" << pt_raw
                          << ", eta=" << eta[i] << "): " << e.what() << ". Setting SF to 1.0." << std::endl;
                sf = 1.0;
            }
            // `sf` corrects the *raw* pt: pt_new = pt_raw * sf. But this column stores a factor
            // that will be applied to the NanoAOD pt downstream (Jet_pt := Jet_pt * factor), so we
            // fold in the raw recovery too:
            //     pt_new = pt_raw * sf = (1 - rawFactor) * Jet_pt * sf
            //   ⇒ factor = pt_new / Jet_pt = (1 - rawFactor) * sf
            factor[i] = (1.0f - rawFactor[i]) * static_cast<float>(sf);
        }
        return factor;
    };

    // NOTE: Type-I MET is no longer propagated here. It is rebuilt from scratch in
    // applyType1MET() (from RawPuppiMET, over Jet + CorrT1METJet, with muon subtraction),
    // which runs after all jet corrections. This function only re-calibrates the jets.
    return df
        .Define("Jet_jecFactor", compute_factor,
                {"year", "Jet_pt", "Jet_eta", "Jet_phi", "Jet_area", "Jet_rawFactor",
                 "Rho_fixedGridRhoFastjetAll", "run"})
        .Redefine("Jet_pt",   "Jet_pt   * Jet_jecFactor")
        .Redefine("Jet_mass", "Jet_mass * Jet_jecFactor")
        // Reset rawFactor so downstream callers see a self-consistent (pt, mass, rawFactor) triple.
        // pt_new * (1 - rawFactor_new) == pt_raw  =>  rawFactor_new = 1 - (1-rawFactor_old)/Jet_jecFactor
        .Redefine("Jet_rawFactor",
                  [](RVec<float> rawFactor, RVec<float> factor) {
                      RVec<float> out(rawFactor.size());
                      for (size_t i = 0; i < rawFactor.size(); ++i) {
                          out[i] = (factor[i] > 0.f)
                                       ? 1.0f - (1.0f - rawFactor[i]) / factor[i]
                                       : rawFactor[i];
                      }
                      return out;
                  },
                  {"Jet_rawFactor", "Jet_jecFactor"});
}

/*
############################################
FAT JET (AK8) ENERGY CORRECTIONS — nominal
############################################

Same recipe as AK4 (raw recovery + L1L2L3Res compound from fatJet_jerc.json.gz, with the
AK8PFPuppi algo). Operates on FatJet_* branches. Does NOT propagate to MET — PuppiMET is
built from AK4 jets only.
*/

RNode applyFatJetEnergyCorrections(const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
                                   const std::unordered_map<std::string, std::string>& jec_prefix_map,
                                   const std::unordered_map<std::string, std::string>& jec_suffix_map,
                                   RNode df, bool isData) {

    auto compute_factor = [cset_jerc, jec_prefix_map, jec_suffix_map, isData](
            std::string year,
            RVec<float> pt, RVec<float> eta, RVec<float> phi, RVec<float> area, RVec<float> rawFactor,
            float rho, unsigned int run) {

        RVec<float> factor(pt.size(), 1.0f);
        if (pt.empty()) return factor;

        auto cs_it = cset_jerc.find(year);
        auto pf_it = jec_prefix_map.find(year);
        auto sf_it = jec_suffix_map.find(year);
        if (cs_it == cset_jerc.end() || pf_it == jec_prefix_map.end() || sf_it == jec_suffix_map.end()) {
            throw std::runtime_error("AK8 JEC: no correction set, prefix, or suffix configured for year '"
                                     + year + "'. Refusing to run with fat jets uncorrected — add this year "
                                       "to the JERC era table in corrections.cpp.");
        }

        const std::string compound_name = makeCompoundJECName(pf_it->second, sf_it->second, isData);
        std::shared_ptr<const correction::CompoundCorrection> compound;
        try {
            compound = cs_it->second.compound().at(compound_name);
        } catch (const std::out_of_range&) {
            throw std::runtime_error("AK8 JEC: compound '" + compound_name
                                     + "' not found in the pinned fatJet_jerc.json.gz for year '" + year
                                     + "'. The JEC tag in the JERC era table (corrections.cpp) is stale relative to the "
                                       "pinned snapshot (version drift). Refusing to run with fat jets uncorrected.");
        }

        for (size_t i = 0; i < pt.size(); ++i) {
            float pt_raw = (1.0f - rawFactor[i]) * pt[i];
            double sf = 1.0;
            try {
                sf = evalJECCompound(*compound, (double)area[i], (double)eta[i], (double)pt_raw,
                                     (double)rho, (double)phi[i], (double)run);
            } catch (const std::exception& e) {
                std::cout << "Warning: JEC evaluation failed for fat jet " << i << " (pt_raw=" << pt_raw
                          << ", eta=" << eta[i] << "): " << e.what() << ". Setting SF to 1.0." << std::endl;
                sf = 1.0;
            }
            factor[i] = (1.0f - rawFactor[i]) * static_cast<float>(sf);
        }
        return factor;
    };

    return df
        .Define("FatJet_jecFactor", compute_factor,
                {"year", "FatJet_pt", "FatJet_eta", "FatJet_phi", "FatJet_area", "FatJet_rawFactor",
                 "Rho_fixedGridRhoFastjetAll", "run"})
        .Redefine("FatJet_pt",   "FatJet_pt   * FatJet_jecFactor")
        .Redefine("FatJet_mass", "FatJet_mass * FatJet_jecFactor")
        // Keep FatJet_rawFactor self-consistent against the new pt/mass.
        .Redefine("FatJet_rawFactor",
                  [](RVec<float> rawFactor, RVec<float> factor) {
                      RVec<float> out(rawFactor.size());
                      for (size_t i = 0; i < rawFactor.size(); ++i) {
                          out[i] = (factor[i] > 0.f)
                                       ? 1.0f - (1.0f - rawFactor[i]) / factor[i]
                                       : rawFactor[i];
                      }
                      return out;
                  },
                  {"FatJet_rawFactor", "FatJet_jecFactor"});
}

/*
############################################
JET ENERGY RESOLUTION — hybrid smearing (MC only)
############################################

Per AK4 jet (using the JEC-corrected pt that this function consumes):
  res = PtResolution(eta, pt, rho)                        // relative resolution
  sf  = ScaleFactor(eta, pt)                              // data/MC width ratio
  unc = SFUncertainty(eta, pt)                            // symmetric ±1σ on sf (split-tag format)
  pt_smear[nom|up|dn] = JERSmear(pt, eta, gen_pt_or_-1, rho, event, res, sf | sf±unc)
               // hybrid: scaling when |pt - pt_gen| < 3*res*pt and gen-matched (genJetIdx>=0),
               // stochastic Gaussian otherwise (same seed for nom/up/dn → fully correlated).
               // Always >= 0.
  pt_new   = pt * pt_smear ; mass_new = mass * pt_smear

Output columns:
  Jet_jerFactor                              nominal smear factor (consumed by applyType1MET)
  Jet_pt / Jet_mass                          redefined to the nominal-smeared values
  with storeVariations (MC + systematics):
  Jet_jerFactor_jerUp / _jerDn               ±1σ smear factors (consumed by applyType1MET)
  Jet_pt_jerUp/Dn, Jet_mass_jerUp/Dn         ±1σ kinematics, built from the same pre-smear
                                             (JEC-corrected) pt/mass as the nominal
*/

RNode applyJetEnergyResolution(const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
                               const std::unordered_map<std::string, correction::CorrectionSet>& cset_jer_smear,
                               const std::unordered_map<std::string, std::string>& jer_res_map,
                               const std::unordered_map<std::string, std::string>& jer_sf_map,
                               const std::unordered_map<std::string, std::string>& jer_unc_map,
                               RNode df, bool storeVariations) {

    auto compute_smear = [cset_jerc, cset_jer_smear, jer_res_map, jer_sf_map, jer_unc_map,
                          storeVariations](
            std::string year,
            RVec<float> pt, RVec<float> eta,
            RVec<short> genJet_idx, RVec<float> genJet_pt, RVec<float> seedpt,
            float rho, unsigned long long event) {

        JERFactors f;
        f.nom.assign(pt.size(), 1.0f);
        f.up.assign(pt.size(), 1.0f);
        f.dn.assign(pt.size(), 1.0f);
        if (pt.empty()) return f;

        auto cs_it = cset_jerc.find(year);
        auto sm_it = cset_jer_smear.find("jer_smear");
        auto rn_it = jer_res_map.find(year);
        auto sn_it = jer_sf_map.find(year);
        auto un_it = jer_unc_map.find(year);
        if (cs_it == cset_jerc.end() || sm_it == cset_jer_smear.end()
            || rn_it == jer_res_map.end() || sn_it == jer_sf_map.end()
            || un_it == jer_unc_map.end()) {
            throw std::runtime_error("AK4 JER: resolution, SF, SF-uncertainty, or smear inputs not "
                                     "configured for year '" + year + "'. Refusing to run with jets "
                                     "un-smeared — add this year to the JERC era table in corrections.cpp.");
        }

        std::shared_ptr<const correction::Correction> res, sf, unc, smear;
        try {
            res   = cs_it->second.at(rn_it->second);
            sf    = cs_it->second.at(sn_it->second);
            unc   = cs_it->second.at(un_it->second);
            smear = sm_it->second.at("JERSmear");
        } catch (const std::out_of_range&) {
            throw std::runtime_error("AK4 JER: a res/sf/unc key (res='" + rn_it->second + "', sf='"
                                     + sn_it->second + "', unc='" + un_it->second
                                     + "') was not found in the pinned jet_jerc.json.gz for year '" + year
                                     + "'. The JER tag in the JERC era table (corrections.cpp) is stale relative to the pinned "
                                       "snapshot (version drift). Refusing to run with jets un-smeared.");
        }

        for (size_t i = 0; i < pt.size(); ++i) {
            const float gen_pt = (genJet_idx[i] >= 0 && genJet_idx[i] < (int)genJet_pt.size())
                                     ? genJet_pt[genJet_idx[i]] : -1.0f;
            const double r = evalJERCorrection(*res, (double)eta[i], (double)pt[i], (double)rho, "nom");
            const double s = evalJERCorrection(*sf,  (double)eta[i], (double)pt[i], (double)rho, "nom");
            const double u = storeVariations
                                 ? evalJERCorrection(*unc, (double)eta[i], (double)pt[i], (double)rho, "nom")
                                 : 0.0;
            const auto sm = evalJERSmearNomUpDn(*smear, (double)pt[i], (double)eta[i], (double)gen_pt,
                                                (double)seedpt[i],
                                                (double)rho, event, r, s, u, storeVariations);
            f.nom[i] = static_cast<float>(sm[0]);
            f.up[i]  = static_cast<float>(sm[1]);
            f.dn[i]  = static_cast<float>(sm[2]);
        }
        return f;
    };

    // NOTE: MET is no longer propagated here. Jet_jerFactor (and the up/dn factors) are
    // consumed by applyType1MET(), which folds JEC*JER into the per-jet correction when
    // rebuilding the Type-I MET from RawPuppiMET. This function only smears the jets.
    // Snapshots of the pre-smear (JEC-corrected) state + matched gen pt: consumed by
    // defineJESVariation, which evaluates the JES shift on the JEC pt and re-applies
    // JER on the shifted pt (official JERC ordering: JES variations BEFORE smearing).
    df = df.Define("_Jet_genpt", matchedGenPt, {"Jet_genJetIdx", "GenJet_pt"})
           .Define("_Jet_pt_prejer",   "Jet_pt")
           .Define("_Jet_mass_prejer", "Jet_mass")
           // Variation-independent PRNG seed for the stochastic branch: the raw pt
           // (rawFactor is kept self-consistent by the JEC step, so this is the NanoAOD raw pt).
           .Define("_Jet_seedpt", "(1.0f - Jet_rawFactor) * Jet_pt");
    df = df.Define("_Jet_jerFactors", compute_smear,
                   {"year", "Jet_pt", "Jet_eta", "Jet_genJetIdx", "GenJet_pt", "_Jet_seedpt",
                    "Rho_fixedGridRhoFastjetAll", "event"});
    df = df.Define("Jet_jerFactor", [](const JERFactors& f) { return f.nom; }, {"_Jet_jerFactors"});
    if (storeVariations) {
        // The pt/mass Defines below run before the nominal Redefine, so they read the
        // pre-smear (JEC-corrected) kinematics — same baseline as the nominal smearing.
        df = df.Define("Jet_jerFactor_jerUp", [](const JERFactors& f) { return f.up; }, {"_Jet_jerFactors"})
               .Define("Jet_jerFactor_jerDn", [](const JERFactors& f) { return f.dn; }, {"_Jet_jerFactors"})
               .Define("Jet_pt_jerUp",   "Jet_pt   * Jet_jerFactor_jerUp")
               .Define("Jet_mass_jerUp", "Jet_mass * Jet_jerFactor_jerUp")
               .Define("Jet_pt_jerDn",   "Jet_pt   * Jet_jerFactor_jerDn")
               .Define("Jet_mass_jerDn", "Jet_mass * Jet_jerFactor_jerDn");
    }
    return df
        .Redefine("Jet_pt",   "Jet_pt   * Jet_jerFactor")
        .Redefine("Jet_mass", "Jet_mass * Jet_jerFactor");
}

/*
############################################
FAT JET (AK8) ENERGY RESOLUTION — hybrid smearing (MC only)
############################################

Same recipe as AK4 (incl. the jerUp/jerDn variation columns) but reading FatJet_* +
GenJetAK8_* and the AK8PFPuppi resolution/SF/SFUncertainty keys. Does NOT propagate
to MET (Type-I MET is built from AK4 only).
*/

RNode applyFatJetEnergyResolution(const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
                                  const std::unordered_map<std::string, correction::CorrectionSet>& cset_jer_smear,
                                  const std::unordered_map<std::string, std::string>& jer_res_map,
                                  const std::unordered_map<std::string, std::string>& jer_sf_map,
                                  const std::unordered_map<std::string, std::string>& jer_unc_map,
                                  RNode df, bool storeVariations) {

    auto compute_smear = [cset_jerc, cset_jer_smear, jer_res_map, jer_sf_map, jer_unc_map,
                          storeVariations](
            std::string year,
            RVec<float> pt, RVec<float> eta,
            RVec<short> genJet_idx, RVec<float> genJet_pt, RVec<float> seedpt,
            float rho, unsigned long long event) {

        JERFactors f;
        f.nom.assign(pt.size(), 1.0f);
        f.up.assign(pt.size(), 1.0f);
        f.dn.assign(pt.size(), 1.0f);
        if (pt.empty()) return f;

        auto cs_it = cset_jerc.find(year);
        auto sm_it = cset_jer_smear.find("jer_smear");
        auto rn_it = jer_res_map.find(year);
        auto sn_it = jer_sf_map.find(year);
        auto un_it = jer_unc_map.find(year);
        if (cs_it == cset_jerc.end() || sm_it == cset_jer_smear.end()
            || rn_it == jer_res_map.end() || sn_it == jer_sf_map.end()
            || un_it == jer_unc_map.end()) {
            throw std::runtime_error("AK8 JER: resolution, SF, SF-uncertainty, or smear inputs not "
                                     "configured for year '" + year + "'. Refusing to run with fat jets "
                                     "un-smeared — add this year to the JERC era table in corrections.cpp.");
        }

        std::shared_ptr<const correction::Correction> res, sf, unc, smear;
        try {
            res   = cs_it->second.at(rn_it->second);
            sf    = cs_it->second.at(sn_it->second);
            unc   = cs_it->second.at(un_it->second);
            smear = sm_it->second.at("JERSmear");
        } catch (const std::out_of_range&) {
            throw std::runtime_error("AK8 JER: a res/sf/unc key (res='" + rn_it->second + "', sf='"
                                     + sn_it->second + "', unc='" + un_it->second
                                     + "') was not found in the pinned fatJet_jerc.json.gz for year '" + year
                                     + "'. The JER tag in the JERC era table (corrections.cpp) is stale relative to the pinned "
                                       "snapshot (version drift). Refusing to run with fat jets un-smeared.");
        }

        for (size_t i = 0; i < pt.size(); ++i) {
            const float gen_pt = (genJet_idx[i] >= 0 && genJet_idx[i] < (int)genJet_pt.size())
                                     ? genJet_pt[genJet_idx[i]] : -1.0f;
            const double r = evalJERCorrection(*res, (double)eta[i], (double)pt[i], (double)rho, "nom");
            const double s = evalJERCorrection(*sf,  (double)eta[i], (double)pt[i], (double)rho, "nom");
            const double u = storeVariations
                                 ? evalJERCorrection(*unc, (double)eta[i], (double)pt[i], (double)rho, "nom")
                                 : 0.0;
            const auto sm = evalJERSmearNomUpDn(*smear, (double)pt[i], (double)eta[i], (double)gen_pt,
                                                (double)seedpt[i],
                                                (double)rho, event, r, s, u, storeVariations);
            f.nom[i] = static_cast<float>(sm[0]);
            f.up[i]  = static_cast<float>(sm[1]);
            f.dn[i]  = static_cast<float>(sm[2]);
        }
        return f;
    };

    // Pre-smear snapshots + matched gen pt for defineJESVariation (see AK4 comment).
    df = df.Define("_FatJet_genpt", matchedGenPt, {"FatJet_genJetAK8Idx", "GenJetAK8_pt"})
           .Define("_FatJet_pt_prejer",   "FatJet_pt")
           .Define("_FatJet_mass_prejer", "FatJet_mass")
           .Define("_FatJet_seedpt", "(1.0f - FatJet_rawFactor) * FatJet_pt");
    df = df.Define("_FatJet_jerFactors", compute_smear,
                   {"year", "FatJet_pt", "FatJet_eta", "FatJet_genJetAK8Idx", "GenJetAK8_pt",
                    "_FatJet_seedpt", "Rho_fixedGridRhoFastjetAll", "event"});
    df = df.Define("FatJet_jerFactor", [](const JERFactors& f) { return f.nom; }, {"_FatJet_jerFactors"});
    if (storeVariations) {
        // Defined before the nominal Redefine → built from the pre-smear (JEC) kinematics.
        df = df.Define("FatJet_jerFactor_jerUp", [](const JERFactors& f) { return f.up; }, {"_FatJet_jerFactors"})
               .Define("FatJet_jerFactor_jerDn", [](const JERFactors& f) { return f.dn; }, {"_FatJet_jerFactors"})
               .Define("FatJet_pt_jerUp",   "FatJet_pt   * FatJet_jerFactor_jerUp")
               .Define("FatJet_mass_jerUp", "FatJet_mass * FatJet_jerFactor_jerUp")
               .Define("FatJet_pt_jerDn",   "FatJet_pt   * FatJet_jerFactor_jerDn")
               .Define("FatJet_mass_jerDn", "FatJet_mass * FatJet_jerFactor_jerDn");
    }
    return df
        .Redefine("FatJet_pt",   "FatJet_pt   * FatJet_jerFactor")
        .Redefine("FatJet_mass", "FatJet_mass * FatJet_jerFactor");
}

/*
############################################
JET ENERGY SCALE — per-source uncertainties (up/down) into suffixed columns
############################################

Per-source JES uncertainty following the official JERC ordering: the shift (1 ± u) is
applied to the JEC-corrected (PRE-smearing) pt, with u(eta, pt) looked up at that JEC pt,
and JER is then RE-EVALUATED on the shifted pt (same event seed as the nominal smearing,
so the variation stays fully correlated). Run *after* applyJet/FatJetEnergyCorrections +
apply*EnergyResolution (it consumes the _<coll>_{pt,mass}_prejer / _<coll>_genpt
snapshots those define). The driver applyJESVariations loops over the 11 Regrouped V2
sources × {Up, Dn} = 22 variations.

  Per jet:  pt_<sfx> = pt_JEC · (1 ± u(eta, pt_JEC)) · c_JER(pt_JEC·(1±u))

  AK4: Jet_pt_<sfx>, Jet_mass_<sfx>, _Jet_var_<sfx>  (full per-jet variation factor
       relative to the JEC pt; consumed by applyType1MET for the per-variation MET)
  AK8: FatJet_pt_<sfx>, FatJet_mass_<sfx>            (FatJet_msoftdrop has its own JEC recipe)
  <sfx> = "jes" + column_label + direction   e.g. "jesAbsoluteUp", "jesAbsoluteYearDn"

Source list lives in kJESRegroupedSources below. Lookup keys in jet_jerc.json.gz are
  <prefix>_<base_source>[_<year_token>]_<suffix>
where year_token comes from jetEnergyCorrections_yearToken when the source is year-
decorrelated (e.g. Regrouped_Absolute_2018 vs the year-correlated Regrouped_Absolute).
*/

namespace {
// Returns a closure giving the per-jet FULL variation factor relative to the JEC
// (pre-smear) pt:  v = (1 + sign·u(eta, pt)) · c_JER(pt·(1+sign·u))
// i.e. JES shift first, then JER smearing re-evaluated at the shifted pt (res, sf and
// the JERSmear gen-match window all see the shifted pt), per the JERC tutorial's
// ordering (JecApplication.cpp::correctedMet). The event seed is shared with the
// nominal smearing, so the stochastic kick is identical and the variation stays fully
// correlated. `gen_pt` is the per-jet matched gen pt (−1 if unmatched).
auto makeJESVarEvaluator(
        const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
        const std::unordered_map<std::string, std::string>& jec_prefix_map,
        const std::unordered_map<std::string, std::string>& jec_suffix_map,
        const std::unordered_map<std::string, std::string>& year_token_map,
        const std::unordered_map<std::string, std::string>& jer_res_map,
        const std::unordered_map<std::string, std::string>& jer_sf_map,
        const std::unordered_map<std::string, correction::CorrectionSet>& cset_jer_smear,
        std::string base_source, bool yearDecorrelated, float sign) {
    return [cset_jerc, jec_prefix_map, jec_suffix_map, year_token_map,
            jer_res_map, jer_sf_map, cset_jer_smear,
            base_source, yearDecorrelated, sign](
            std::string year, RVec<float> pt, RVec<float> eta, RVec<float> gen_pt,
            RVec<float> seed_pt, float rho, unsigned long long event) {
        RVec<float> out(pt.size(), 1.0f);
        if (pt.empty()) return out;
        auto cs_it = cset_jerc.find(year);
        auto pf_it = jec_prefix_map.find(year);
        auto sf_it = jec_suffix_map.find(year);
        auto rn_it = jer_res_map.find(year);
        auto sn_it = jer_sf_map.find(year);
        auto sm_it = cset_jer_smear.find("jer_smear");
        if (cs_it == cset_jerc.end() || pf_it == jec_prefix_map.end() || sf_it == jec_suffix_map.end()
            || rn_it == jer_res_map.end() || sn_it == jer_sf_map.end() || sm_it == cset_jer_smear.end()) {
            throw std::runtime_error("JES uncertainty: correction set / prefix / suffix / JER inputs not "
                                     "configured for year '" + year + "' (source '" + base_source + "'). "
                                     "Refusing to emit an empty JES variation — add this year to the JERC "
                                     "era table in corrections.cpp.");
        }
        std::string src = base_source;
        if (yearDecorrelated) {
            auto yt_it = year_token_map.find(year);
            if (yt_it == year_token_map.end()) {
                throw std::runtime_error("JES uncertainty: no year token configured for year '" + year
                                         + "' (needed for year-decorrelated source '" + base_source
                                         + "'). Add it to the JERC era table in corrections.cpp.");
            }
            src += "_" + yt_it->second;
        }
        const std::string unc_name = pf_it->second + "_" + src + "_" + sf_it->second;
        std::shared_ptr<const correction::Correction> unc, res, sf, smear;
        try {
            unc   = cs_it->second.at(unc_name);
            res   = cs_it->second.at(rn_it->second);
            sf    = cs_it->second.at(sn_it->second);
            smear = sm_it->second.at("JERSmear");
        } catch (const std::out_of_range&) {
            throw std::runtime_error("JES uncertainty: source '" + unc_name
                                     + "' (or its JER res/sf partners) not found in the pinned jerc file "
                                       "for year '" + year
                                     + "'. The JEC tag in the JERC era table (corrections.cpp) is stale relative to the "
                                       "pinned snapshot (version drift). Refusing to emit an empty variation.");
        }
        for (size_t i = 0; i < pt.size(); ++i) {
            const double u  = unc->evaluate({(double)eta[i], (double)pt[i]});
            const double s  = 1.0 + (double)sign * u;
            const double ptS = (double)pt[i] * s;
            const double r   = evalJERCorrection(*res, (double)eta[i], ptS, (double)rho, "nom");
            const double sfv = evalJERCorrection(*sf,  (double)eta[i], ptS, (double)rho, "nom");
            // Window + raw-pt stochastic seeding handled by the helper (same seed as the
            // nominal smearing → same Gaussian → variation fully correlated).
            const auto sm = evalJERSmearNomUpDn(*smear, ptS, (double)eta[i], (double)gen_pt[i],
                                                (double)seed_pt[i], (double)rho, event,
                                                r, sfv, 0.0, false);
            out[i] = static_cast<float>(s * sm[0]);
        }
        return out;
    };
}

// Defines one JES variation for one collection ("Jet" or "FatJet"): the per-jet full
// variation factor _<collection>_var_<sfx> (relative to the pre-smear JEC pt) and the
// varied <collection>_pt_<sfx> / <collection>_mass_<sfx>, built from the pre-smear
// snapshots (_<coll>_{pt,mass}_prejer, _<coll>_genpt) taken in apply*EnergyResolution.
// The correction set and algo suffix are looked up in the registry from the collection
// name; AK4 and AK8 share the same JEC release (prefix) and year tokens, so the JES
// sources are correlated between the two algorithms per the JERC prescription.
//
// For the main jets, the matching met_pt_<sfx> / met_phi_<sfx> are built in
// applyType1MET(), which reads _Jet_var_<sfx> and rebuilds the Type-I MET from
// RawPuppiMET for this variation. (NOT FatJet_msoftdrop — that has its own JEC recipe;
// AK8 does not enter the Type-I MET, which is built from AK4 only.)
RNode defineJESVariation(RNode df, const std::string& collection,
                         const std::string& column_label, const std::string& base_source,
                         bool yearDecorrelated, const std::string& direction) {
    if (direction != "Up" && direction != "Dn") return df;
    const bool isFat = (collection == "FatJet");
    const float sign = (direction == "Up") ? +1.0f : -1.0f;
    const std::string sfx = "jes" + column_label + direction;
    const std::string varCol = "_" + collection + "_var_" + sfx;

    auto eval_factor = makeJESVarEvaluator(
        isFat ? fatJetEnergyCorrections() : jetEnergyCorrections(),
        jetEnergyCorrections_JEC_prefix(),   // same JEC release for AK4 and AK8
        isFat ? fatJetEnergyCorrections_JEC_suffix() : jetEnergyCorrections_JEC_suffix(),
        jetEnergyCorrections_yearToken(),
        isFat ? fatJetEnergyResolution_JER_res_name() : jetEnergyResolution_JER_res_name(),
        isFat ? fatJetEnergyResolution_JER_sf_name()  : jetEnergyResolution_JER_sf_name(),
        jetEnergyResolution_smear(),
        base_source, yearDecorrelated, sign);

    const std::string pre = "_" + collection;
    return df
        .Define(varCol, eval_factor,
                {"year", pre + "_pt_prejer", collection + "_eta", pre + "_genpt",
                 pre + "_seedpt", "Rho_fixedGridRhoFastjetAll", "event"})
        .Define(collection + "_pt_"   + sfx, pre + "_pt_prejer   * " + varCol)
        .Define(collection + "_mass_" + sfx, pre + "_mass_prejer * " + varCol);
}
struct JESSourceSpec {
    const char* label;          // column-name suffix, e.g. "Absolute" or "AbsoluteYear"
    const char* base_source;    // JEC source key root, e.g. "Regrouped_Absolute"
    bool yearDecorrelated;      // append _<year_token> at lookup time?
};

// 11 Regrouped V2 sources — confirmed against jet_jerc.json.gz / fatJet_jerc.json.gz
// for all 9 campaigns on 2026-05-14. JME-POG standard set;
static const std::vector<JESSourceSpec> kJESRegroupedSources = {
    {"Absolute",            "Regrouped_Absolute",       false},
    {"BBEC1",               "Regrouped_BBEC1",          false},
    {"EC2",                 "Regrouped_EC2",            false},
    {"HF",                  "Regrouped_HF",             false},
    {"RelativeBal",         "Regrouped_RelativeBal",    false},
    {"FlavorQCD",           "Regrouped_FlavorQCD",      false},
    {"AbsoluteYear",        "Regrouped_Absolute",       true},
    {"BBEC1Year",           "Regrouped_BBEC1",          true},
    {"EC2Year",             "Regrouped_EC2",            true},
    {"HFYear",              "Regrouped_HF",             true},
    {"RelativeSampleYear",  "Regrouped_RelativeSample", true},
};
static const std::array<const char*, 2> kJESDirections = {"Up", "Dn"};

// MET unclustered energy (UES) variation table, pairing the NanoAOD branch and our
// output suffix that follows this framework's Up/Dn convention.
struct UESVariationSpec {
    const char* suffix;     // our column suffix, e.g. "metunclUp" → met_pt_metunclUp
    const char* nanoToken;  // NanoAOD branch token, e.g. "UnclusteredUp" → PuppiMET_ptUnclusteredUp
};
static const std::array<UESVariationSpec, 2> kUESVariations = {{
    {"metunclUp", "UnclusteredUp"},
    {"metunclDn", "UnclusteredDown"},
}};
} // anonymous namespace

RNode applyJESVariations(RNode df) {
    for (const auto& src : kJESRegroupedSources) {
        for (const char* dir : kJESDirections) {
            df = defineJESVariation(df, "Jet",    src.label, src.base_source, src.yearDecorrelated, dir);
            df = defineJESVariation(df, "FatJet", src.label, src.base_source, src.yearDecorrelated, dir);
        }
    }
    return df;
}

static bool g_storeSysts = false;
void setStoreSysts(bool v) { g_storeSysts = v; }

// DEBUGGING ONLY — see the warning on setApplyJetVetoMaps in corrections.h.
static bool g_applyJetVetoMaps = true;
void setApplyJetVetoMaps(bool v) { g_applyJetVetoMaps = v; }
bool applyJetVetoMapsEnabled() { return g_applyJetVetoMaps; }

std::vector<std::string> jesVariationSuffixes() {
    if (!g_storeSysts) return {};
    std::vector<std::string> out;
    out.reserve(kJESRegroupedSources.size() * kJESDirections.size());
    for (const auto& src : kJESRegroupedSources) {
        for (const char* dir : kJESDirections) {
            out.push_back(std::string("jes") + src.label + dir);
        }
    }
    return out;
}

std::vector<std::string> jerVariationSuffixes() {
    if (!g_storeSysts) return {};
    return {"jerUp", "jerDn"};
}

std::vector<std::string> kinematicVariationSuffixes() {
    auto out = jesVariationSuffixes();
    auto jer = jerVariationSuffixes();
    out.insert(out.end(), jer.begin(), jer.end());
    return out;
}

std::vector<std::string> unclusteredVariationSuffixes() {
    if (!g_storeSysts) return {};
    std::vector<std::string> out;
    out.reserve(kUESVariations.size());
    for (const auto& v : kUESVariations) out.push_back(v.suffix);
    return out;
}

// Every suffix for which met_pt_<sfx> / met_phi_<sfx> may exist: the kinematic (JES+JER)
// ones written by applyType1MET plus the MET-only UES ones. Deliberately NOT merged into
// kinematicVariationSuffixes() — that list drives the per-variation JET columns and the
// channel-pass flags in selections.cpp, which UES has none of.
/*
############################################
TYPE-1 PUPPI MET — full rebuild from RawPuppiMET
############################################

Matches the central JERC application tutorial (cms-analysis/jme/jerc-application-tutorial,
JecApplication.cpp::correctedMet + JecApplication.py::_recompute_type1_met) and the JERC
Type-1 MET recipe (Type1MET.pdf, 2026-04-17):

  MET_T1 = RawPuppiMET + Σ_jets  p⃗_raw,musub · (c_L1 − c_full)

summed over the *merged* AK4 list (Jet + CorrT1METJet). For PUPPI the L1FastJet offset
c_L1 ≡ 1.0 (verified for AK4PFPuppi across all 9 eras in the pinned 2026-06-05 files), so
the per-jet contribution is  p⃗_raw,musub · (1 − c_full).

Per jet:
  * p_raw       = (1 − rawFactor) · pt              (CorrT1METJet ships rawPt directly)
  * p_raw,musub = p_raw · (1 − muonSubtrFactor)     direction = phi + muonSubtrDeltaPhi
  * c_full      = c_JEC · c_JER · c_JES-shift        (c_JEC = L1L2L3Res compound at p_raw,musub)
  * selection   : (chEmEF+neEmEF) < 0.9, |eta| < 5.2, and c_full · p_raw,musub > 15 GeV
  * baseline    : RawPuppiMET (NOT PuppiMET — PuppiMET already has Type-1 applied)

Decisions / scope (documented for the review):
  - c_L1 = 1 hardcoded (PUPPI). If a future file gives L1 ≠ 1, this must evaluate L1FastJet.
    (The tutorial evaluates L1FastJet explicitly; identical to c_L1=1 for PUPPI.)
  - JEC compound evaluated at the muon-subtracted pt (matches the tutorial's per-level chain,
    which feeds ptRawMinusMuon through L1·L2·L3). Muon-jets are the only ones affected.
  - Direction = phi + muonSubtrDeltaPhi, per Type1MET.pdf §1.5 (accounts for the jet's
    direction change when the muon is removed). NOTE: the JERC tutorial code and getT1CHSMET
    use the plain jet phi here; we deliberately follow the PDF instead.
  - BOTH collections (Jet and CorrT1METJet) carry the full JEC·JER·JES-shift per scenario —
    no approximation. CorrT1METJet get their own gen-matched JER (ΔR<0.2 to GenJet, then
    JERSmear) and their own per-source JES shift, exactly like the main jets.
      · main-jet JER = the analysis Jet_jerFactor (evaluated on the JEC pt, consistent with
        the jets used in the event selection);
      · CorrT1METJet JER = gen-matched JERSmear on their JEC muon-subtracted pt (they have no
        Jet_genJetIdx, so we ΔR-match to GenJet as the JERC tutorial does).
    JES shift is evaluated at the JEC·JER-corrected pt for both (nominal-JER × per-source
    shift), matching this framework's main-jet convention.
  - MET-φ corrections (Run 2) are applied afterwards, on top of this rebuilt MET.
*/

namespace {

// Per-(main-)jet building blocks for the Type-1 MET sum. Computed once, reused by the
// nominal MET and every JES variation (which only swap the per-jet shift factor).
struct T1JetBlocks {
    RVec<double> rmpx, rmpy;   // raw muon-subtracted momentum comps (direction = phi + muonSubtrDeltaPhi)
    RVec<double> corrjec;      // JEC factor (L1L2L3Res compound at the muon-subtracted pt)
    RVec<double> rawmusubpt;   // |raw muon-subtracted pT|, for the 15 GeV threshold
    RVec<unsigned char> pass;  // selection: (neEmEF + chEmEF) < 0.9 and |eta| < 5.2
};

// If column `src` exists, alias `out`->`src`; otherwise Define `out` as an RVec<float> of
// `fill`, length matching `sizeRef`. Lets the rebuild tolerate NanoAOD versions lacking
// muonSubtrDeltaPhi / CorrT1METJet_EmEF, and the data path lacking Jet_jerFactor.
RNode aliasOrFill(RNode df, const std::string& out, const std::string& src,
                  const std::string& sizeRef, float fill) {
    auto cols = df.GetColumnNames();
    if (std::find(cols.begin(), cols.end(), src) != cols.end())
        return df.Alias(out, src);
    return df.Define(out, [fill](const RVec<float>& ref) { return RVec<float>(ref.size(), fill); },
                     {sizeRef});
}

} // anonymous namespace

RNode applyType1MET(RNode df, bool isData) {
    // Tolerate optional / MC-only branches.
    df = aliasOrFill(df, "_t1_jet_musubDphi", "Jet_muonSubtrDeltaPhi",          "Jet_eta",          0.0f);
    df = aliasOrFill(df, "_t1_jer",           "Jet_jerFactor",                  "Jet_eta",          1.0f);
    df = aliasOrFill(df, "_t1_jer_jerUp",     "Jet_jerFactor_jerUp",            "Jet_eta",          1.0f);
    df = aliasOrFill(df, "_t1_jer_jerDn",     "Jet_jerFactor_jerDn",            "Jet_eta",          1.0f);
    df = aliasOrFill(df, "_t1_ct1_musubDphi", "CorrT1METJet_muonSubtrDeltaPhi", "CorrT1METJet_eta", 0.0f);
    df = aliasOrFill(df, "_t1_ct1_EmEF",      "CorrT1METJet_EmEF",              "CorrT1METJet_eta", 0.0f);
    // Single EM-fraction column so the block builder is shared with CorrT1METJet.
    df = df.Define("_t1_jet_emfrac", "Jet_neEmEF + Jet_chEmEF");

    // --- shared per-jet block builder (used for both Jet and CorrT1METJet) ---------------
    auto build_blocks = [isData](
            std::string year,
            const RVec<float>& rawpt, const RVec<float>& eta, const RVec<float>& phi,
            const RVec<float>& area,  const RVec<float>& musubF, const RVec<float>& musubDphi,
            const RVec<float>& emFrac,
            float rho, unsigned int run) {
        T1JetBlocks b;
        const size_t n = rawpt.size();
        b.rmpx.resize(n); b.rmpy.resize(n); b.corrjec.resize(n);
        b.rawmusubpt.resize(n); b.pass.resize(n);
        if (n == 0) return b;
        const auto& cs = jetEnergyCorrections().at(year);
        const std::string cname = makeCompoundJECName(jetEnergyCorrections_JEC_prefix().at(year),
                                                      jetEnergyCorrections_JEC_suffix().at(year), isData);
        auto comp = cs.compound().at(cname);
        for (size_t i = 0; i < n; ++i) {
            const double rawmusubpt = (double)rawpt[i] * (1.0 - (double)musubF[i]);
            // MET-shift direction = phi + muonSubtrDeltaPhi, per Type1MET.pdf §1.5 (accounts
            // for the jet's direction change when the muon is removed; falls back to phi when
            // the branch is absent, via the 0-filled musubDphi column).
            const double dir = (double)phi[i] + (double)musubDphi[i];
            b.rmpx[i] = rawmusubpt * std::cos(dir);
            b.rmpy[i] = rawmusubpt * std::sin(dir);
            b.rawmusubpt[i] = rawmusubpt;
            double sf = 1.0;
            try {
                // JEC chain evaluated on the muon-subtracted pt (JERC tutorial JecApplication.cpp).
                sf = evalJECCompound(*comp, (double)area[i], (double)eta[i], rawmusubpt,
                                     (double)rho, (double)phi[i], (double)run);
            } catch (const std::exception& e) {
                std::cout << "Warning: JEC evaluation failed for MET jet " << i << " (musub_pt=" << rawmusubpt
                          << ", eta=" << eta[i] << "): " << e.what() << ". Setting SF to 1.0." << std::endl;
                sf = 1.0;
            }
            b.corrjec[i] = sf;
            b.pass[i] = (emFrac[i] < 0.9f && std::abs((double)eta[i]) < 5.2) ? 1 : 0;
        }
        return b;
    };
    // Handle Jet collection
    df = df.Define("_t1_jet_blocks", build_blocks,
                   {"year", "_Jet_rawpt", "Jet_eta", "Jet_phi", "Jet_area",
                    "Jet_muonSubtrFactor", "_t1_jet_musubDphi", "_t1_jet_emfrac",
                    "Rho_fixedGridRhoFastjetAll", "run"});
    // Handle CorrT1METJet collection (already raw pt, no rawFactor)
    df = df.Define("_t1_ct1_blocks", build_blocks,
                   {"year", "CorrT1METJet_rawPt", "CorrT1METJet_eta", "CorrT1METJet_phi",
                    "CorrT1METJet_area", "CorrT1METJet_muonSubtrFactor", "_t1_ct1_musubDphi",
                    "_t1_ct1_EmEF", "Rho_fixedGridRhoFastjetAll", "run"});

    // Nominal per-jet "shift" = 1, sized to each collection.
    auto ones_like = [](const T1JetBlocks& b) { return RVec<float>(b.corrjec.size(), 1.0f); };
    df = df.Define("_t1_jet_ones", ones_like, {"_t1_jet_blocks"});
    df = df.Define("_t1_ct1_ones", ones_like, {"_t1_ct1_blocks"});

    // --- CorrT1METJet JER: gen-matched (ΔR<0.2 to GenJet) JERSmear on the JEC musub pt -----
    // (main-jet JER is the analysis Jet_jerFactor, already in _t1_jer). Same nom/up/dn
    // triplet as the main jets (sf ± SFUncertainty, shared seed). Data → ones.
    if (!isData) {
        const bool storeVariations = g_storeSysts;
        auto build_ct1_jer = [storeVariations](std::string year, const T1JetBlocks& b,
                                const RVec<float>& eta, const RVec<float>& phi,
                                const RVec<float>& gpt, const RVec<float>& geta, const RVec<float>& gphi,
                                float rho, unsigned long long event) {
            JERFactors f;
            f.nom.assign(b.corrjec.size(), 1.0f);
            f.up.assign(b.corrjec.size(), 1.0f);
            f.dn.assign(b.corrjec.size(), 1.0f);
            if (b.corrjec.empty()) return f;
            std::shared_ptr<const correction::Correction> res, sf, unc;
            std::shared_ptr<const correction::Correction> smear;
            try {
                res   = jetEnergyCorrections().at(year).at(jetEnergyResolution_JER_res_name().at(year));
                sf    = jetEnergyCorrections().at(year).at(jetEnergyResolution_JER_sf_name().at(year));
                unc   = jetEnergyCorrections().at(year).at(jetEnergyResolution_JER_unc_name().at(year));
                smear = jetEnergyResolution_smear().at("jer_smear").at("JERSmear");
            } catch (const std::out_of_range&) {
                throw std::runtime_error("CorrT1METJet JER: res/sf/unc/smear key missing for year '" + year
                                         + "'. Stale JER tag vs pinned snapshot (see the JERC era table in corrections.cpp).");
            }
            for (size_t i = 0; i < b.corrjec.size(); ++i) {
                const double pt_jec = b.rawmusubpt[i] * b.corrjec[i];   // JEC-corrected musub pt
                // Nearest GenJet within ΔR<0.2 (CorrT1METJet has no Jet_genJetIdx).
                float gen_pt = -1.0f; double best = 0.2 * 0.2;
                for (size_t g = 0; g < gpt.size(); ++g) {
                    const double dphi = std::atan2(std::sin((double)phi[i] - (double)gphi[g]),
                                                   std::cos((double)phi[i] - (double)gphi[g]));
                    const double deta = (double)eta[i] - (double)geta[g];
                    const double dr2  = deta * deta + dphi * dphi;
                    if (dr2 < best) { best = dr2; gen_pt = gpt[g]; }
                }
                const double r = evalJERCorrection(*res, (double)eta[i], pt_jec, (double)rho, "nom");
                const double s = evalJERCorrection(*sf,  (double)eta[i], pt_jec, (double)rho, "nom");
                const double u = storeVariations
                                     ? evalJERCorrection(*unc, (double)eta[i], pt_jec, (double)rho, "nom")
                                     : 0.0;
                const auto sm = evalJERSmearNomUpDn(*smear, pt_jec, (double)eta[i], (double)gen_pt,
                                                    b.rawmusubpt[i],
                                                    (double)rho, event, r, s, u, storeVariations);
                f.nom[i] = static_cast<float>(sm[0]);
                f.up[i]  = static_cast<float>(sm[1]);
                f.dn[i]  = static_cast<float>(sm[2]);
            }
            return f;
        };
        df = df.Define("_t1_ct1_jerFactors", build_ct1_jer,
                       {"year", "_t1_ct1_blocks", "CorrT1METJet_eta", "CorrT1METJet_phi",
                        "GenJet_pt", "GenJet_eta", "GenJet_phi", "Rho_fixedGridRhoFastjetAll", "event"});
        df = df.Define("_t1_ct1_jer", [](const JERFactors& f) { return f.nom; }, {"_t1_ct1_jerFactors"});
        if (g_storeSysts) {
            df = df.Define("_t1_ct1_jer_jerUp", [](const JERFactors& f) { return f.up; }, {"_t1_ct1_jerFactors"})
                   .Define("_t1_ct1_jer_jerDn", [](const JERFactors& f) { return f.dn; }, {"_t1_ct1_jerFactors"});
        }
    } else {
        df = df.Alias("_t1_ct1_jer", "_t1_ct1_ones");
    }

    // CorrT1METJet JEC-corrected (pre-JER) pt — the baseline at which the JES variation
    // factor is evaluated (u at the JEC pt, then JER re-evaluated on the shifted pt),
    // matching the JERC ordering used for the main jets in defineJESVariation.
    df = df.Define("_t1_ct1_jecpt",
                   [](const T1JetBlocks& b) {
                       RVec<float> o(b.corrjec.size());
                       for (size_t i = 0; i < b.corrjec.size(); ++i)
                           o[i] = static_cast<float>(b.rawmusubpt[i] * b.corrjec[i]);
                       return o;
                   },
                   {"_t1_ct1_blocks"});

    // Per-source CorrT1METJet JES variation factors (same 22 Regrouped sources as the
    // main jets, same JES-before-JER ordering): v = (1 ± u(eta, pt_JEC)) · c_JER(pt_JEC·(1±u)).
    if (!isData && g_storeSysts) {
        // Per-jet matched gen pt (ΔR<0.2 to GenJet) for the re-evaluated JER inside the
        // variation — same matching as build_ct1_jer above.
        df = df.Define("_t1_ct1_genpt",
                       [](const RVec<float>& eta, const RVec<float>& phi,
                          const RVec<float>& gpt, const RVec<float>& geta, const RVec<float>& gphi) {
                           RVec<float> out(eta.size(), -1.0f);
                           for (size_t i = 0; i < eta.size(); ++i) {
                               double best = 0.2 * 0.2;
                               for (size_t g = 0; g < gpt.size(); ++g) {
                                   const double dphi = std::atan2(std::sin((double)phi[i] - (double)gphi[g]),
                                                                  std::cos((double)phi[i] - (double)gphi[g]));
                                   const double deta = (double)eta[i] - (double)geta[g];
                                   const double dr2  = deta * deta + dphi * dphi;
                                   if (dr2 < best) { best = dr2; out[i] = gpt[g]; }
                               }
                           }
                           return out;
                       },
                       {"CorrT1METJet_eta", "CorrT1METJet_phi", "GenJet_pt", "GenJet_eta", "GenJet_phi"});
        df = df.Define("_t1_ct1_seedpt",
                       [](const T1JetBlocks& b) {
                           RVec<float> o(b.rawmusubpt.size());
                           for (size_t i = 0; i < b.rawmusubpt.size(); ++i)
                               o[i] = static_cast<float>(b.rawmusubpt[i]);
                           return o;
                       },
                       {"_t1_ct1_blocks"});
        for (const auto& src : kJESRegroupedSources) {
            for (const char* dir : kJESDirections) {
                const float sign = (std::string(dir) == "Up") ? +1.0f : -1.0f;
                auto eval = makeJESVarEvaluator(jetEnergyCorrections(),
                                                jetEnergyCorrections_JEC_prefix(),
                                                jetEnergyCorrections_JEC_suffix(),
                                                jetEnergyCorrections_yearToken(),
                                                jetEnergyResolution_JER_res_name(),
                                                jetEnergyResolution_JER_sf_name(),
                                                jetEnergyResolution_smear(),
                                                src.base_source, src.yearDecorrelated, sign);
                df = df.Define("_t1_ct1_var_jes" + std::string(src.label) + dir, eval,
                               {"year", "_t1_ct1_jecpt", "CorrT1METJet_eta", "_t1_ct1_genpt",
                                "_t1_ct1_seedpt", "Rho_fixedGridRhoFastjetAll", "event"});
            }
        }
    }

    // --- assemble a MET for one scenario: sum both collections, corr_full = corrjec*jer*shift ---
    auto met_from = [](const T1JetBlocks& jb, const RVec<float>& jjer, const RVec<float>& jshift,
                       const T1JetBlocks& cb, const RVec<float>& cjer, const RVec<float>& cshift,
                       float rawmet_pt, float rawmet_phi) {
        double px = (double)rawmet_pt * std::cos((double)rawmet_phi);
        double py = (double)rawmet_pt * std::sin((double)rawmet_phi);
        auto add = [&px, &py](const T1JetBlocks& b, const RVec<float>& jer, const RVec<float>& shift) {
            for (size_t i = 0; i < b.corrjec.size(); ++i) {
                if (!b.pass[i]) continue;
                const double jf = (i < jer.size())   ? (double)jer[i]   : 1.0;
                const double sh = (i < shift.size()) ? (double)shift[i] : 1.0;
                const double corr_full = b.corrjec[i] * jf * sh;         // c_L1 = 1 (PUPPI)
                if (corr_full * b.rawmusubpt[i] <= 15.0) continue;       // Type-I threshold on full pT
                px += b.rmpx[i] * (1.0 - corr_full);
                py += b.rmpy[i] * (1.0 - corr_full);
            }
        };
        add(jb, jjer, jshift);
        add(cb, cjer, cshift);
        return std::make_pair(static_cast<float>(std::hypot(px, py)),
                              static_cast<float>(std::atan2(py, px)));
    };

    // Nominal MET (JEC·JER on both collections; JER=1 in data via the _t1_*_ones fallbacks).
    df = df.Define("_t1_met_nom", met_from,
                   {"_t1_jet_blocks", "_t1_jer", "_t1_jet_ones",
                    "_t1_ct1_blocks", "_t1_ct1_jer", "_t1_ct1_ones",
                    "RawPuppiMET_pt", "RawPuppiMET_phi"});
    df = df.Define("met_pt",  "(float)_t1_met_nom.first");
    df = df.Define("met_phi", "(float)_t1_met_nom.second");

    // Per-variation MET.
    if (!isData) {
        // JES: swap in the full per-source variation factor (JES shift + re-evaluated
        // JER, relative to the JEC pt) for BOTH collections; the nominal-JER column is
        // replaced by ones since the variation factor already carries its own JER.
        for (const auto& sfx : jesVariationSuffixes()) {
            const std::string metCol = "_t1_met_" + sfx;
            df = df.Define(metCol, met_from,
                           {"_t1_jet_blocks", "_t1_jet_ones", "_Jet_var_" + sfx,
                            "_t1_ct1_blocks", "_t1_ct1_ones", "_t1_ct1_var_" + sfx,
                            "RawPuppiMET_pt", "RawPuppiMET_phi"});
            df = df.Define("met_pt_"  + sfx, "(float)" + metCol + ".first");
            df = df.Define("met_phi_" + sfx, "(float)" + metCol + ".second");
        }
        // JER: swap in the ±1σ smear factor for BOTH collections (JES shift stays 1).
        for (const auto& sfx : jerVariationSuffixes()) {
            const std::string metCol = "_t1_met_" + sfx;
            df = df.Define(metCol, met_from,
                           {"_t1_jet_blocks", "_t1_jer_" + sfx, "_t1_jet_ones",
                            "_t1_ct1_blocks", "_t1_ct1_jer_" + sfx, "_t1_ct1_ones",
                            "RawPuppiMET_pt", "RawPuppiMET_phi"});
            df = df.Define("met_pt_"  + sfx, "(float)" + metCol + ".first");
            df = df.Define("met_phi_" + sfx, "(float)" + metCol + ".second");
        }
    }
    return df;
}

/*
############################################
MET UNCLUSTERED ENERGY (UES) VARIATIONS
############################################

MET = -Σ p⃗_T over all PF candidates, and for systematics that sum splits into two disjoint
pieces. The clustered part carries JES/JER, already propagated by
applyType1MET. The unclustered remainder has a scale uncertainty of its own, computed in CMSSW by
pat::MET::shiftedP4(UnclusteredEnUp/Down).

Being disjoint from the jet sum also makes the shift JEC-invariant, which is what lets it
be lifted from NanoAOD as a vector and added to our rebuilt Type-1 MET.
Taking the difference is what cancels nano's JEC:

  dx± = PuppiMET_ptUnclustered±·cos(PuppiMET_phiUnclustered±) - PuppiMET_pt·cos(PuppiMET_phi)
  dy± = PuppiMET_ptUnclustered±·sin(PuppiMET_phiUnclustered±) - PuppiMET_pt·sin(PuppiMET_phi)
  px± = met_pt·cos(met_phi) + dx±,    py± = met_pt·sin(met_phi) + dy±

Runs after applyType1MET and before applyMETPhiCorrections, so met_pt/met_phi here are the
pre-φ-corrected Type-1 rebuild.
*/

RNode applyMETUnclusteredVariations(RNode df, bool isData) {
    if (isData) return df;                                  // MC-only nuisance
    if (unclusteredVariationSuffixes().empty()) return df;   // nominal-only (no --systs)

    for (const char* b : {"PuppiMET_pt", "PuppiMET_phi",
                          "PuppiMET_ptUnclusteredUp",   "PuppiMET_phiUnclusteredUp",
                          "PuppiMET_ptUnclusteredDown", "PuppiMET_phiUnclusteredDown"}) {
        if (!df.HasColumn(b))
            throw std::runtime_error(
                std::string("MET unclustered (UES): input is missing branch '") + b
                + "'. Either the input predates v15 or the skim dropped the branches. "
                  "Drop --systs if a nominal-only production is really what is wanted.");
    }

    // One lambda for both directions: the nano direction is carried by the input column
    // names, so Up and Dn are produced by identical code.
    auto shift_met = [](float met_pt, float met_phi,
                        float nano_pt, float nano_phi,
                        float nano_pt_var, float nano_phi_var) {
        const double dx = (double)nano_pt_var * std::cos((double)nano_phi_var)
                        - (double)nano_pt     * std::cos((double)nano_phi);
        const double dy = (double)nano_pt_var * std::sin((double)nano_phi_var)
                        - (double)nano_pt     * std::sin((double)nano_phi);
        const double px = (double)met_pt * std::cos((double)met_phi) + dx;
        const double py = (double)met_pt * std::sin((double)met_phi) + dy;
        return std::make_pair(static_cast<float>(std::hypot(px, py)),
                              static_cast<float>(std::atan2(py, px)));
    };

    for (const auto& v : kUESVariations) {
        const std::string sfx    = v.suffix;
        const std::string metCol = "_uncl_met_" + sfx;
        df = df.Define(metCol, shift_met,
                       {"met_pt", "met_phi", "PuppiMET_pt", "PuppiMET_phi",
                        std::string("PuppiMET_pt")  + v.nanoToken,
                        std::string("PuppiMET_phi") + v.nanoToken});
        df = df.Define("met_pt_"  + sfx, "(float)" + metCol + ".first");
        df = df.Define("met_phi_" + sfx, "(float)" + metCol + ".second");
    }
    return df;
}

/*
############################################
MET PHI CORRECTIONS
############################################
*/

RNode applyMETPhiCorrections(RNode df, bool isData) {
    auto eval_correction = [isData] (std::string year, float pt, float phi, unsigned char npvs, unsigned int run) {
        double pt_corr = pt;
        double phi_corr = phi;

        if (metCorrections().find(year) == metCorrections().end()) {
            static std::unordered_set<std::string> warned_years;
            if (warned_years.find(year) == warned_years.end()) {
                std::cout << "Warning: MET correction set for year " << year << " not found. Skipping MET phi corrections." << std::endl;
                warned_years.insert(year);
            }
            return std::make_pair(static_cast<float>(pt_corr), static_cast<float>(phi_corr));
        }

        std::string pt_corr_name = isData ? "pt_metphicorr_puppimet_data" : "pt_metphicorr_puppimet_mc";
        std::string phi_corr_name = isData ? "phi_metphicorr_puppimet_data" : "phi_metphicorr_puppimet_mc";

        // Correction json is only defined for up to 6500, so do not pass larger values
        float pt_max = 6499.9;
        float pt_to_pass = std::min(pt,pt_max);
        pt_corr = metCorrections().at(year).at(pt_corr_name)->evaluate({pt_to_pass, phi, static_cast<double>(npvs), static_cast<double>(run)});
        phi_corr = metCorrections().at(year).at(phi_corr_name)->evaluate({pt_to_pass, phi, static_cast<double>(npvs), static_cast<double>(run)});

        return std::make_pair(static_cast<float>(pt_corr), static_cast<float>(phi_corr));
    };
    // Runs after applyType1MET + applyMETUnclusteredVariations, so it refines the
    // already-rebuilt MET rather than starting from PuppiMET.
    // TO CHECK IF THIS IS CORRECT
    //
    // It is added to each varied column rather than corrected once on the nominal and
    // inherited, because the variations are not built from the nominal: applyType1MET
    // rebuilds each JES/JER MET independently from RawPuppiMET with its own jet factors,
    // so met_pt_<sfx> never references met_pt. Only the UES columns are nominal + delta,
    // and they are built before this step. Adding a translation to each column is anyway
    // identical to displacing a corrected nominal: (nom - d) + (var - nom) = var - d.
    auto eval_shift = [eval_correction](std::string year, unsigned char npvs, unsigned int run) {
        const auto probe = eval_correction(year, 0.f, 0.f, npvs, run);
        return std::make_pair(
            static_cast<float>((double)probe.first * std::cos((double)probe.second)),
            static_cast<float>((double)probe.first * std::sin((double)probe.second)));
    };
    df = df.Define("_MET_xyshift", eval_shift, {"year", "PV_npvs", "run"});

    auto apply_shift = [](float pt, float phi, const std::pair<float, float> &s) {
        const double px = (double)pt * std::cos((double)phi) + (double)s.first;
        const double py = (double)pt * std::sin((double)phi) + (double)s.second;
        return std::make_pair(static_cast<float>(std::hypot(px, py)),
                              static_cast<float>(std::atan2(py, px)));
    };
    auto correct_one = [&apply_shift](RNode d, const std::string& sfx) {
        const std::string tail   = sfx.empty() ? std::string{} : "_" + sfx;
        const std::string ptCol  = "met_pt"  + tail;
        const std::string phiCol = "met_phi" + tail;
        const std::string helper = "_MET_phicorr" + tail;
        d = d.Define(helper, apply_shift, {ptCol, phiCol, "_MET_xyshift"});
        return d.Redefine(ptCol,  helper + ".first")
                .Redefine(phiCol, helper + ".second");
    };

    df = correct_one(df, "");

    std::vector<std::string> metSuffixes;
    for (auto&& col : df.GetDefinedColumnNames()) {
        if (!col.starts_with("met_pt_")) continue;
        const std::string sfx = std::string(col).substr(sizeof("met_pt_") - 1);
        if (df.HasColumn("met_phi_" + sfx)) metSuffixes.push_back(sfx);
    }
    for (const auto& sfx : metSuffixes) df = correct_one(df, sfx);
    return df;
}

/*
############################################
HEM Corrections
############################################
*/

RNode HEMCorrection(RNode df, bool isData) {
    auto HEMCorrections = [isData](unsigned int run, unsigned long long event, std::string sample_year, RVec<float> pt, RVec<float> eta, RVec<float> phi, RVec<float> jet_id) {
        RVec<bool> jet_mask;
        if (sample_year == "2018" && ((isData && run >= 319077) || (!isData && event % 100 < 64))) {
            jet_mask = (jet_id >= 2 && pt > 15.0); // v>= 30 skims follow NanoAOD jetID convention https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#Jets
                                                    // works also for v < 30 skims, which set jetId==3 : "pass tight ID, fail tightLepVeto", jetId==7 : "pass tight and tightLepVeto ID"
            auto eta_ = eta[jet_mask];
            auto phi_ = phi[jet_mask];
            for (size_t i = 0; i < eta_.size(); i++) {
                if (eta_[i] > -3.2 && eta_[i] < -1.3 && phi_[i] > -1.57 && phi_[i] < -0.87) {
                    return false;
                }
            }
        }
        return true;
    };

    return df.Define("HEMVeto", HEMCorrections, {"run", "event", "year", "Jet_pt", "Jet_eta", "Jet_phi", "Jet_jetId"}).Filter("HEMVeto");
}

/*
############################################
ELECTRON SCALE AND SMEARING CORRECTIONS
############################################
*/

RNode applyElectronScaleAndSmearing(RNode df, bool isData) {
    auto eval_data_scale = [](std::string year, RVec<float> pt, RVec<float> eta, RVec<float> deltaEtaSC, RVec<float> r9, RVec<unsigned char> seedGain, unsigned int run) {
        RVec<float> scales(pt.size(), 1.0);
        if (electronSSCorrections().find(year) == electronSSCorrections().end()) {
            static std::unordered_set<std::string> warned_years;
            if (warned_years.find(year) == warned_years.end()) {
                std::cout << "Warning: Electron SS correction set for year " << year << " not found. Skipping scale corrections." << std::endl;
                warned_years.insert(year);
            }
            return scales;
        }

        for (size_t i = 0; i < pt.size(); i++) {
            float scEta = eta[i] + deltaEtaSC[i];
            scales[i] = electronSSCorrections().at(year).compound().at("Scale")->evaluate({"scale", static_cast<double>(run), scEta, r9[i], pt[i], static_cast<double>(seedGain[i])});
        }
        return scales;
    };

    auto eval_data_smearWidth = [](std::string year, RVec<float> pt, RVec<float> eta, RVec<float> deltaEtaSC, RVec<float> r9, RVec<float> scales) {
        RVec<float> widths(pt.size(), 0.0);
        if (electronSSCorrections().find(year) == electronSSCorrections().end()) {
            return widths;
        }

        for (size_t i = 0; i < pt.size(); i++) {
            float scEta = eta[i] + deltaEtaSC[i];
            float pt_corr = pt[i] * scales[i];
            widths[i] = electronSSCorrections().at(year).at("SmearAndSyst")->evaluate({"smear", pt_corr, r9[i], scEta});
        }
        return widths;
    };

    auto eval_mc_smear_factor = [](std::string year, RVec<float> pt, RVec<float> eta, RVec<float> deltaEtaSC, RVec<float> r9, unsigned long long event) {
        RVec<float> smear_factors(pt.size(), 1.0);
        if (electronSSCorrections().find(year) == electronSSCorrections().end()) {
            static std::unordered_set<std::string> warned_years;
            if (warned_years.find(year) == warned_years.end()) {
                std::cout << "Warning: Electron SS correction set for year " << year << " not found. Skipping smearing corrections." << std::endl;
                warned_years.insert(year);
            }
            return smear_factors;
        }

        TRandom3 rng(event);
        for (size_t i = 0; i < pt.size(); i++) {
            float scEta = eta[i] + deltaEtaSC[i];
            float smear = electronSSCorrections().at(year).at("SmearAndSyst")->evaluate({"smear", pt[i], r9[i], scEta});
            smear_factors[i] = 1.0 + smear * rng.Gaus(0.0, 1.0);
        }
        return smear_factors;
    };

    auto eval_mc_smearWidth = [](std::string year, RVec<float> pt, RVec<float> eta, RVec<float> deltaEtaSC, RVec<float> r9) {
        RVec<float> widths(pt.size(), 0.0);
        if (electronSSCorrections().find(year) == electronSSCorrections().end()) {
            return widths;
        }

        for (size_t i = 0; i < pt.size(); i++) {
            float scEta = eta[i] + deltaEtaSC[i];
            widths[i] = electronSSCorrections().at(year).at("SmearAndSyst")->evaluate({"smear", pt[i], r9[i], scEta});
        }
        return widths;
    };

    auto calc_energyErr = [](RVec<float> pt, RVec<float> eta, RVec<float> energyErr, RVec<float> widths, RVec<float> factors) {
        RVec<float> new_energyErr(pt.size(), 0.0);
        for (size_t i = 0; i < pt.size(); i++) {
            float energy = pt[i] * TMath::CosH(eta[i]);
            new_energyErr[i] = TMath::Sqrt(energyErr[i] * energyErr[i] + (energy * widths[i]) * (energy * widths[i])) * factors[i];
        }
        return new_energyErr;
    };

    if (isData) {
        return df.Define("_ele_scale", eval_data_scale, {"year", "Electron_pt", "Electron_eta", "Electron_deltaEtaSC", "Electron_r9", "Electron_seedGain", "run"})
                 .Define("_ele_smearWidth", eval_data_smearWidth, {"year", "Electron_pt", "Electron_eta", "Electron_deltaEtaSC", "Electron_r9", "_ele_scale"})
                 .Redefine("Electron_energyErr", calc_energyErr, {"Electron_pt", "Electron_eta", "Electron_energyErr", "_ele_smearWidth", "_ele_scale"})
                 .Redefine("Electron_pt", "Electron_pt * _ele_scale")
                 .Redefine("Electron_mass", "Electron_mass * _ele_scale");
    } else {
        return df.Define("_ele_smearFactor", eval_mc_smear_factor, {"year", "Electron_pt", "Electron_eta", "Electron_deltaEtaSC", "Electron_r9", "event"})
                 .Define("_ele_smearWidth", eval_mc_smearWidth, {"year", "Electron_pt", "Electron_eta", "Electron_deltaEtaSC", "Electron_r9"})
                 .Redefine("Electron_energyErr", calc_energyErr, {"Electron_pt", "Electron_eta", "Electron_energyErr", "_ele_smearWidth", "_ele_smearFactor"})
                 .Redefine("Electron_pt", "Electron_pt * _ele_smearFactor")
                 .Redefine("Electron_mass", "Electron_mass * _ele_smearFactor");
    }
}

/*
############################################
JET VETO MAPS
############################################
*/

/*
The per-jet gate differs between the two runs, following the JME prescriptions.
Both ask for pT > 15 GeV and chEmEF + neEmEF < 0.9; they differ in the jet ID
and in the muon-overlap treatment:

  Run 3 : tightLepVeto jet ID.
  Run 2 : tight jet ID, and jets overlapping a PF muon within dR < 0.2 are
          exempt from the veto.

The ID cuts are written `>= 6` / `>= 2` rather than `== 6` / `== 2` so that both
jetId encodings work: v >= 30 skims use the NanoAOD convention 2*tight +
4*tightLepVeto (values 0/2/6), older skims set 3 = "tight, fails tightLepVeto"
and 7 = "tight and tightLepVeto".
*/
RNode applyJetVetoMaps(RNode df) {
    auto eval_correction = [] (std::string year, bool isRun2, RVec<float> pt, RVec<float> eta, RVec<float> phi, RVec<float> jet_id, RVec<float> jet_nuEmEF, RVec<float> jet_chEmEF, RVec<float> muon_eta, RVec<float> muon_phi, RVec<bool> muon_isPFcand) {
        RVec<bool> jet_veto_map;

        // Hard-fail on an unknown era, matching the JEC/JER behaviour. This used to warn once
        // and flag nothing, which silently produced un-vetoed output. The trap is real: the
        // 2022/2023 samples are tagged "2022"/"2023", which match none of the four split era
        // keys the maps are published under ("2022Re-recoBCD", "2022Re-recoE+PromptFG",
        // "2023PromptC", "2023PromptD"), so they would run with no veto at all. The maps for
        // those eras ARE wired in eraVetoMapTable() and ready to use -- what is missing is the
        // era split in the sample metadata, which must be fixed when the 2022/2023 skims land
        // (the dataset names carry the era: Run2022C/D -> BCD, Run2022E/F/G -> E+PromptFG,
        // Run2023C -> PromptC, Run2023D -> PromptD).
        if (jetVetoMaps().find(year) == jetVetoMaps().end()) {
            std::string known;
            for (const auto& [k, _] : eraVetoMapTable()) known += (known.empty() ? "" : ", ") + k;
            throw std::runtime_error(
                "Jet veto map: no map configured for year '" + year + "'. Known eras: " + known
                + ". A year that reaches this point would run with NO veto map applied, so it is a "
                  "hard error rather than a warning. If this is a 2022/2023 sample, re-tag its "
                  "metadata to the matching split era key (the plain '2022'/'2023' keys are not "
                  "valid -- the maps are published per run era).");
        }

        const float min_jet_id = isRun2 ? 2.0f : 6.0f;   // tight (Run 2) vs tightLepVeto (Run 3)

        for (size_t i = 0; i < eta.size(); i++) {
            if (!(pt[i] > 15.0 && jet_id[i] >= min_jet_id && (jet_nuEmEF[i] + jet_chEmEF[i]) < 0.9)) {
                jet_veto_map.push_back(false);
                continue;
            }
            // Run 2 only: the recipe only checks jets that do not overlap a PF muon within dR < 0.2.
            if (isRun2) {
                bool overlaps_pf_muon = false;
                for (size_t j = 0; j < muon_eta.size(); j++) {
                    if (muon_isPFcand[j] && ROOT::VecOps::DeltaR(eta[i], muon_eta[j], phi[i], muon_phi[j]) < 0.2) {
                        overlaps_pf_muon = true;
                        break;
                    }
                }
                if (overlaps_pf_muon) {
                    jet_veto_map.push_back(false);
                    continue;
                }
            }
            float eta_ = eta[i];
            if (std::abs(eta_) > 5.19) {
                eta_ = 5.19 * (eta_ > 0 ? 1 : -1);
            }
            jet_veto_map.push_back(jetVetoMaps().at(year).at(jetVetoMap_names().at(year))->evaluate({"jetvetomap", eta_, phi[i]}) != 0);
        }

        return jet_veto_map;
    };

    return df.Define("Jet_vetoMap", eval_correction, {"year", "isRun2", "Jet_pt", "Jet_eta", "Jet_phi", "Jet_jetId", "Jet_neEmEF", "Jet_chEmEF", "Muon_eta", "Muon_phi", "Muon_isPFcand"});
}

/*
############################################
JET MASS SCALE (JMS) AND RESOLUTION (JMR) — generic, per-mass-column
############################################

Column-agnostic helpers: pass the target mass branch (and, for JMR, the gen
reference mass + gen-index columns) as arguments so the same code serves
`FatJet_msoftdrop`, the GloParT regressed mass, or any other analysis mass
variable.

  JMS  : m' = max(0, m + Δshift[year]) // additive shift in GeV
  JMR  : matched   →  m' = max(0, m_gen + f * (m - m_gen))
         unmatched →  m' = max(0, m * (1 + √max(f²-1,0) * σrel[year] * N(0,1)))

For msd, the POG-standard proxy is `GenJetAK8_mass[FatJet_genJetAK8Idx]`.

Currently unwired — this analysis calibrates JMS/JMR on the GloParT mass, not
on msd, and the GloParT calibration hasn't been derived yet.
When it lands, wire the calls into `applyMCCorrections` with the
GloParT column names + per-era shift/factor/σ_rel maps from the calibration
fit. Up/down systematic variants add a `variation` arg carrying the ±1σ
shift/scale.
*/

RNode applyJetMassScale(const std::unordered_map<std::string, float>& shift_map,
                     const std::string& mass_col,
                     RNode df) {
    auto eval = [shift_map, mass_col](std::string year, RVec<float> m) {
        auto it = shift_map.find(year);
        if (it == shift_map.end()) {
            static std::unordered_set<std::string> warned;
            if (warned.insert(mass_col + "/" + year).second) {
                std::cout << "Warning: mass-scale shift not defined for year " << year
                          << " on column " << mass_col << ". Skipping." << std::endl;
            }
            return m;
        }
        const float shift = it->second;
        if (shift == 0.0f || m.empty()) return m;
        for (auto& x : m) x = std::max(0.0f, x + shift);
        return m;
    };
    return df.Redefine(mass_col, eval, {"year", mass_col});
}

RNode applyJetMassResolution(const std::unordered_map<std::string, float>& factor_map,
                          const std::unordered_map<std::string, float>& sigma_rel_map,
                          const std::string& mass_col,
                          const std::string& gen_mass_col,
                          const std::string& gen_idx_col,
                          RNode df) {
    auto eval = [factor_map, sigma_rel_map, mass_col](std::string year,
                             RVec<float> m,
                             RVec<short> gen_idx, RVec<float> gen_mass,
                             unsigned int lumi, unsigned long long event) {
        auto it = factor_map.find(year);
        if (it == factor_map.end()) {
            static std::unordered_set<std::string> warned;
            if (warned.insert(mass_col + "/" + year).second) {
                std::cout << "Warning: mass-resolution factor not defined for year " << year
                          << " on column " << mass_col << ". Skipping." << std::endl;
            }
            return m;
        }
        const float f = it->second;
        if (f == 1.0f || m.empty()) return m;

        const float widen = std::sqrt(std::max(f * f - 1.0f, 0.0f));
        auto sr_it = sigma_rel_map.find(year);
        const float sigma_rel = (sr_it != sigma_rel_map.end()) ? sr_it->second : 1.0f;
        TRandom3 rng((static_cast<unsigned long long>(lumi) << 10) ^ event);
        for (size_t i = 0; i < m.size(); ++i) {
            const bool matched = (gen_idx[i] >= 0
                                  && gen_idx[i] < static_cast<int>(gen_mass.size()));
            if (matched) {
                const float m_gen = gen_mass[gen_idx[i]];
                m[i] = std::max(0.0f, m_gen + f * (m[i] - m_gen));
            } else {
                const float kick = static_cast<float>(rng.Gaus(0.0, 1.0));
                m[i] = std::max(0.0f, m[i] * (1.0f + widen * sigma_rel * kick));
            }
        }
        return m;
    };
    return df.Redefine(mass_col, eval,
                       {"year", mass_col, gen_idx_col, gen_mass_col,
                        "luminosityBlock", "event"});
}

/*
############################################
GENERAL CORRECTIONS
############################################
*/

// Preserve the raw jet kinematics before JEC/JER rescale the stored columns. Both consumers
// run later in the graph and cannot recover these values themselves: JEC keeps *_rawFactor
// consistent with the new pt/mass, but JER rescales pt and mass without updating it, so from
// that point on (1 - rawFactor)*x is the raw value times the smear factor, not the raw value.
//   _Jet_rawpt      -> applyType1MET, which rebuilds the MET sum from raw AK4 jets.
//   _FatJet_rawmass -> AK8JetsSelection, which builds the GloParT regressed mass.
RNode snapshotRawJetKinematics(RNode df) {
    return df.Define("_Jet_rawpt",      "(1.0f - Jet_rawFactor) * Jet_pt")
             .Define("_FatJet_rawmass", "(1.0f - FatJet_rawFactor) * FatJet_mass");
}

RNode applyDataCorrections(RNode df_) {
    auto df = snapshotRawJetKinematics(df_);
    // Re-apply the latest JEC stack (raw recovery + L1*L2*L3*Residual compound). No MET
    // propagation here — Type-I MET is rebuilt from RawPuppiMET in applyType1MET below.
    df = applyJetEnergyCorrections(jetEnergyCorrections(),
                                   jetEnergyCorrections_JEC_prefix(),
                                   jetEnergyCorrections_JEC_suffix(),
                                   df, /*isData=*/true);
    df = applyFatJetEnergyCorrections(fatJetEnergyCorrections(),
                                      jetEnergyCorrections_JEC_prefix(),   // same JEC release for AK4 and AK8
                                      fatJetEnergyCorrections_JEC_suffix(),
                                      df, /*isData=*/true);
    // Rebuild Type-I MET (nominal only for data), then apply the Run 2 MET-φ correction on top.
    df = applyType1MET(df, /*isData=*/true);
    df = applyMETPhiCorrections(df, true);
    df = HEMCorrection(df, true);
    df = applyElectronScaleAndSmearing(df, true);
    return df;
}

RNode applyMCCorrections(RNode df_) {
    auto df = snapshotRawJetKinematics(df_);
    // Nominal JEC for AK4 and AK8 (no MET propagation here).
    df = applyJetEnergyCorrections(jetEnergyCorrections(),
                                   jetEnergyCorrections_JEC_prefix(),
                                   jetEnergyCorrections_JEC_suffix(),
                                   df, /*isData=*/false);
    df = applyFatJetEnergyCorrections(fatJetEnergyCorrections(),
                                      jetEnergyCorrections_JEC_prefix(),   // same JEC release for AK4 and AK8
                                      fatJetEnergyCorrections_JEC_suffix(),
                                      df, /*isData=*/false);
    // JER smearing on top of JEC-corrected jets (defines Jet_jerFactor and, with systematics
    // enabled, the jerUp/jerDn factor + kinematic branches — all consumed by applyType1MET).
    df = applyJetEnergyResolution(jetEnergyCorrections(),
                                  jetEnergyResolution_smear(),
                                  jetEnergyResolution_JER_res_name(),
                                  jetEnergyResolution_JER_sf_name(),
                                  jetEnergyResolution_JER_unc_name(),
                                  df, /*storeVariations=*/g_storeSysts);
    df = applyFatJetEnergyResolution(fatJetEnergyCorrections(),
                                     jetEnergyResolution_smear(),
                                     fatJetEnergyResolution_JER_res_name(),
                                     fatJetEnergyResolution_JER_sf_name(),
                                     fatJetEnergyResolution_JER_unc_name(),
                                     df, /*storeVariations=*/g_storeSysts);
    // JES uncertainties — 11 Regrouped V2 sources × {Up, Dn} as suffixed branches.
    // Must run after the nominal JEC+JER (consumes their pre-smear snapshots; the shift is
    // applied on the JEC pt with JER re-evaluated on the shifted pt, per the JERC ordering)
    // (defines _Jet_var_<sfx>, consumed by applyType1MET for the per-variation MET).
    if (g_storeSysts) df = applyJESVariations(df);
    // Rebuild Type-I MET from RawPuppiMET over Jet + CorrT1METJet, for the nominal and every
    // JES/JER variation.
    df = applyType1MET(df, /*isData=*/false);
    // UES ±1σ on top of the rebuilt Type-I MET
    df = applyMETUnclusteredVariations(df, /*isData=*/false);
    df = applyMETPhiCorrections(df, false);
    // GloParT JMS/JMR would be wired here via applyJetMassScale / applyJetMassResolution
    // once the calibration is derived. FatJet_msoftdrop is intentionally not calibrated
    // (loose object cut only; GloParT is used in the rest of the analysis).
    df = HEMCorrection(df, false);
    df = applyElectronScaleAndSmearing(df, false);
    return df;
}
