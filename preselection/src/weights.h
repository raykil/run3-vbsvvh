#ifndef WEIGHTS
#define WEIGHTS

#pragma once

#include <unordered_map>
#include <string>
#include <iostream>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RVec.hxx"

#include "utils.h"
#include "correction.h"

using correction::CorrectionSet;
using RNode = ROOT::RDF::RNode;
using ROOT::VecOps::RVec;

/*
############################################
GOLDEN JSON
############################################
*/
RNode applyGoldenJSONWeight(const lumiMask& golden, RNode df);
const auto LumiMask = lumiMask::fromJSON({
    "corrections/goldenJson/Cert_Collisions2022_355100_362760_Golden.json",
    "corrections/goldenJson/Cert_Collisions2023_366442_370790_Golden.json",
    "corrections/goldenJson/Cert_Collisions2024_378981_386951_Golden.json",
    "corrections/goldenJson/Cert_Collisions2025_391658_398903_Golden.json",
    "corrections/goldenJson/Cert_271036-284044_13TeV_Legacy2016_Collisions16.json",
    "corrections/goldenJson/Cert_294927-306462_13TeV_UL2017_Collisions17_Golden.json",
    "corrections/goldenJson/Cert_314472-325175_13TeV_Legacy2018_Collisions18.json"
});

/*
############################################
PILEUP SFs
############################################
*/
const std::unordered_map<std::string, correction::CorrectionSet> pileupScaleFactors = {
    {"2016preVFP", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2016preVFP-UL-NanoAODv9/latest/puWeights.json.gz")},
    {"2016postVFP", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2016postVFP-UL-NanoAODv9/latest/puWeights.json.gz")},
    {"2017", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2017-UL-NanoAODv9/latest/puWeights.json.gz")},
    {"2018", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2018-UL-NanoAODv9/latest/puWeights.json.gz")},
    {"2022Re-recoBCD", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-22CDSep23-Summer22-NanoAODv12/latest/puWeights.json.gz")},
    {"2022Re-recoE+PromptFG", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-22EFGSep23-Summer22EE-NanoAODv12/latest/puWeights.json.gz")},
    {"2023PromptC", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-23CSep23-Summer23-NanoAODv12/latest/puWeights.json.gz")},
    {"2023PromptD", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-23DSep23-Summer23BPix-NanoAODv12/latest/puWeights.json.gz")},
    {"2024Prompt", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/puWeights_BCDEFGHI.json.gz")},
    {"2025", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-25Prompt-Summer24-NanoAODv15/latest/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz")}
};
const std::unordered_map<std::string, std::string> pileupScaleFactors_yearmap = {
    {"2016preVFP", "Collisions16_UltraLegacy_goldenJSON"},
    {"2016postVFP", "Collisions16_UltraLegacy_goldenJSON"},
    {"2017", "Collisions17_UltraLegacy_goldenJSON"},
    {"2018", "Collisions18_UltraLegacy_goldenJSON"},
    {"2022Re-recoBCD", "Collisions2022_355100_357900_eraBCD_GoldenJson"},
    {"2022Re-recoE+PromptFG", "Collisions2022_359022_362760_eraEFG_GoldenJson"},
    {"2023PromptC", "Collisions2023_366403_369802_eraBC_GoldenJson"},
    {"2023PromptD", "Collisions2023_369803_370790_eraD_GoldenJson"},
    {"2024Prompt", "Collisions24_BCDEFGHI_goldenJSON"},
    {"2025", "Collisions25_goldenJSON"}
};
RNode applyPileupScaleFactors(std::unordered_map<std::string, correction::CorrectionSet> cset_pileup, std::unordered_map<std::string, std::string> year_map, RNode df);


/*
############################################
Lepton SFs wrapper (putting e and m together)
############################################
*/

RNode lepSFWrapper(RNode df,
    bool isData,
    const std::string& ele_sf_name,
    const std::string& muo_sf_name,
    bool include_trigger_sf = false,
    const std::string& ele_output_name = "weightsyst_eleSF",
    const std::string& muo_output_name = "weightsyst_muoSF");

/*
############################################
MUON SFs
############################################
*/

struct MuonCorrectionSet {
    correction::CorrectionSet cset;
    bool abs_eta; // true = use abs(eta), false = use signed eta
};

const std::unordered_map<std::string, std::vector<std::string>> muonWorkingPointSFs = {
    {"_weight_muon_looseid_looseiso", {
        "_weight_muon_reco",
        "_weight_muon_id_loose",
        "_weight_muon_iso_looseid_looseiso"
    }},
    {"_weight_muon_mediumid_tightiso", {
        "_weight_muon_reco",
        "_weight_muon_id_medium",
        "_weight_muon_iso_mediumid_tightiso"
    }},
    {"_weight_muon_tightid_tightiso", {
        "_weight_muon_reco",
        "_weight_muon_id_tight",
        "_weight_muon_iso_tightid_tightiso"
    }}
};

const std::unordered_map<std::string, MuonCorrectionSet> muonScaleFactors = {
    {"2016preVFP",            {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2016preVFP-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"),                       true}},  // abs eta
    {"2016postVFP",           {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2016postVFP-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"),                      true}},  // abs eta
    {"2017",                  {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2017-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"),                             true}},  // abs eta
    {"2018",                  {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2018-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"),                             true}},  // abs eta
    {"2022Re-recoBCD",        {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/2026-06-18/muon_Z.json.gz"),                  true}},  // abs eta
    {"2022Re-recoE+PromptFG", {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22EFGSep23-Summer22EE-NanoAODv12/2026-06-18/muon_Z.json.gz"),               true}},  // abs eta
    {"2023PromptC",           {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23CSep23-Summer23-NanoAODv12/2026-06-18/muon_Z.json.gz"),                   false}}, // signed eta
    {"2023PromptD",           {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23DSep23-Summer23BPix-NanoAODv12/2026-06-18/muon_Z.json.gz"),               false}}, // signed eta
    {"2024Prompt",            {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-06-18/muon_Z.json.gz"),false}}, // signed eta
    {"2025",                  {*CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-25Prompt-Summer24-NanoAODv15/2026-04-28/muon_Z.json.gz"),                   false}}, // signed eta
};

struct MuonSFYear {
    std::string correction_key;
    float pt_min; // minimum valid pt, verified against JSON bin edges
};

struct MuonSFConfig {
    std::unordered_map<std::string, MuonSFYear> year_map;
};

const std::unordered_map<std::string, MuonSFConfig> muonSFConfigs = {
    // RECO SF (Run 2 only, only defined above 40 GeV, returns 1.0 for Run 3 years)
    {"_weight_muon_reco", {{
        {"2016preVFP",  {"NUM_TrackerMuons_DEN_genTracks", 40.0}},
        {"2016postVFP", {"NUM_TrackerMuons_DEN_genTracks", 40.0}},
        {"2017",        {"NUM_TrackerMuons_DEN_genTracks", 40.0}},
        {"2018",        {"NUM_TrackerMuons_DEN_genTracks", 40.0}},
    }}},
    // ID SFs
    {"_weight_muon_id_loose", {{
        {"2016preVFP",            {"NUM_LooseID_DEN_TrackerMuons", 15.1}},
        {"2016postVFP",           {"NUM_LooseID_DEN_TrackerMuons", 15.1}},
        {"2017",                  {"NUM_LooseID_DEN_TrackerMuons", 15.1}},
        {"2018",                  {"NUM_LooseID_DEN_TrackerMuons", 15.1}},
        {"2022Re-recoBCD",        {"NUM_LooseID_DEN_TrackerMuons", 15.1}},
        {"2022Re-recoE+PromptFG", {"NUM_LooseID_DEN_TrackerMuons", 15.1}},
        {"2023PromptC",           {"NUM_LooseID_DEN_TrackerMuons", 15.1}},
        {"2023PromptD",           {"NUM_LooseID_DEN_TrackerMuons", 15.1}},
        {"2024Prompt",            {"NUM_LooseID_DEN_TrackerMuons", 10.1}},
        {"2025",                  {"NUM_LooseID_DEN_TrackerMuons", 10.1}},
    }}},
    {"_weight_muon_id_medium", {{
        {"2016preVFP",            {"NUM_MediumID_DEN_TrackerMuons", 15.1}},
        {"2016postVFP",           {"NUM_MediumID_DEN_TrackerMuons", 15.1}},
        {"2017",                  {"NUM_MediumID_DEN_TrackerMuons", 15.1}},
        {"2018",                  {"NUM_MediumID_DEN_TrackerMuons", 15.1}},
        {"2022Re-recoBCD",        {"NUM_MediumID_DEN_TrackerMuons", 15.1}},
        {"2022Re-recoE+PromptFG", {"NUM_MediumID_DEN_TrackerMuons", 15.1}},
        {"2023PromptC",           {"NUM_MediumID_DEN_TrackerMuons", 15.1}},
        {"2023PromptD",           {"NUM_MediumID_DEN_TrackerMuons", 15.1}},
        {"2024Prompt",            {"NUM_MediumID_DEN_TrackerMuons", 10.1}},
        {"2025",                  {"NUM_MediumID_DEN_TrackerMuons", 10.1}},
    }}},
    {"_weight_muon_id_tight", {{
        {"2016preVFP",            {"NUM_TightID_DEN_TrackerMuons", 15.1}},
        {"2016postVFP",           {"NUM_TightID_DEN_TrackerMuons", 15.1}},
        {"2017",                  {"NUM_TightID_DEN_TrackerMuons", 15.1}},
        {"2018",                  {"NUM_TightID_DEN_TrackerMuons", 15.1}},
        {"2022Re-recoBCD",        {"NUM_TightID_DEN_TrackerMuons", 15.1}},
        {"2022Re-recoE+PromptFG", {"NUM_TightID_DEN_TrackerMuons", 15.1}},
        {"2023PromptC",           {"NUM_TightID_DEN_TrackerMuons", 15.1}},
        {"2023PromptD",           {"NUM_TightID_DEN_TrackerMuons", 15.1}},
        {"2024Prompt",            {"NUM_TightID_DEN_TrackerMuons", 10.1}},
        {"2025",                  {"NUM_TightID_DEN_TrackerMuons", 10.1}},
    }}},
    // ISO SFs
    {"_weight_muon_iso_looseid_looseiso", {{
        {"2016preVFP",            {"NUM_LooseRelIso_DEN_LooseID", 15.1}},
        {"2016postVFP",           {"NUM_LooseRelIso_DEN_LooseID", 15.1}},
        {"2017",                  {"NUM_LooseRelIso_DEN_LooseID", 15.1}},
        {"2018",                  {"NUM_LooseRelIso_DEN_LooseID", 15.1}},
        {"2022Re-recoBCD",        {"NUM_LoosePFIso_DEN_LooseID",  15.1}},
        {"2022Re-recoE+PromptFG", {"NUM_LoosePFIso_DEN_LooseID",  15.1}},
        {"2023PromptC",           {"NUM_LoosePFIso_DEN_LooseID",  15.1}},
        {"2023PromptD",           {"NUM_LoosePFIso_DEN_LooseID",  15.1}},
        {"2024Prompt",            {"NUM_LoosePFIso_DEN_LooseID",  10.1}},
        {"2025",                  {"NUM_LoosePFIso_DEN_LooseID",  10.1}},
    }}},
    {"_weight_muon_iso_mediumid_tightiso", {{
        {"2016preVFP",            {"NUM_TightRelIso_DEN_MediumID", 15.1}},
        {"2016postVFP",           {"NUM_TightRelIso_DEN_MediumID", 15.1}},
        {"2017",                  {"NUM_TightRelIso_DEN_MediumID", 15.1}},
        {"2018",                  {"NUM_TightRelIso_DEN_MediumID", 15.1}},
        {"2022Re-recoBCD",        {"NUM_TightPFIso_DEN_MediumID",  15.1}},
        {"2022Re-recoE+PromptFG", {"NUM_TightPFIso_DEN_MediumID",  15.1}},
        {"2023PromptC",           {"NUM_TightPFIso_DEN_MediumID",  15.1}},
        {"2023PromptD",           {"NUM_TightPFIso_DEN_MediumID",  15.1}},
        {"2024Prompt",            {"NUM_TightPFIso_DEN_MediumID",  10.1}},
        {"2025",                  {"NUM_TightPFIso_DEN_MediumID",  10.1}},
    }}},
    {"_weight_muon_iso_tightid_tightiso", {{
        {"2016preVFP",            {"NUM_TightRelIso_DEN_TightIDandIPCut", 15.1}},
        {"2016postVFP",           {"NUM_TightRelIso_DEN_TightIDandIPCut", 15.1}},
        {"2017",                  {"NUM_TightRelIso_DEN_TightIDandIPCut", 15.1}},
        {"2018",                  {"NUM_TightRelIso_DEN_TightIDandIPCut", 15.1}},
        {"2022Re-recoBCD",        {"NUM_TightPFIso_DEN_TightID",          15.1}},
        {"2022Re-recoE+PromptFG", {"NUM_TightPFIso_DEN_TightID",          15.1}},
        {"2023PromptC",           {"NUM_TightPFIso_DEN_TightID",          15.1}},
        {"2023PromptD",           {"NUM_TightPFIso_DEN_TightID",          15.1}},
        {"2024Prompt",            {"NUM_TightPFIso_DEN_TightID",          10.1}},
        {"2025",                  {"NUM_TightPFIso_DEN_TightID",          10.1}},
    }}},
    {"_weight_muon_trigger", {{
        {"2016preVFP",            {"NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  30.0}},
        {"2016postVFP",           {"NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",  30.0}},
        {"2017",                  {"NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",               30.0}},
        {"2018",                  {"NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",               30.0}},
        {"2022Re-recoBCD",        {"NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",               30.0}},
        {"2022Re-recoE+PromptFG", {"NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",               30.0}},
        {"2023PromptC",           {"NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",               30.0}},
        {"2023PromptD",           {"NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",               30.0}},
        //{"2024Prompt",            {"NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",               30.0}}, // This key is not in the 2024 json, skipping for now
    }}},
};

// Function declarations
RNode applyMuonScaleFactors(std::unordered_map<std::string, MuonCorrectionSet> cset_muon, std::string output_name, MuonSFConfig config, RNode df);
RNode combineScaleFactorWeightsByKey(RNode df, std::string output_name, std::vector<std::string> input_keys);
RNode applyMuonWorkingPointSFs(RNode df, bool isData, std::vector<std::string> wp_keys);

/*
############################################
ELECTRON SFs
############################################
*/
const std::unordered_map<std::string, correction::CorrectionSet> electronScaleFactors = {
    {"2016preVFP",            *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2016preVFP-UL-NanoAODv15/2025-12-05/electron.json.gz")},
    {"2016postVFP",           *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2016postVFP-UL-NanoAODv15/2025-12-05/electron.json.gz")},
    {"2017",                  *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2017-UL-NanoAODv15/2025-12-05/electron.json.gz")},
    {"2018",                  *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2018-UL-NanoAODv15/2025-12-05/electron.json.gz")},
    {"2022Re-recoBCD",        *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22CDSep23-Summer22-NanoAODv12/2025-12-15/electron.json.gz")},
    {"2022Re-recoE+PromptFG", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-12-15/electron.json.gz")},
    {"2023PromptC",           *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23CSep23-Summer23-NanoAODv12/2025-12-15/electron.json.gz")},
    {"2023PromptD",           *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-12-15/electron.json.gz")},
    {"2024Prompt",            *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-12-15/electron.json.gz")},
    {"2025",                  *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-25Prompt-Summer24-NanoAODv15/2026-06-26/electron.json.gz")},
};

// The lowest-pt electron reco working point, per era. EGM provides RecoBelow20 (valid
// from 10 GeV) everywhere except 2025, where the lowest available bin is Reco20to75 and
// sub-20 GeV electrons have to be clamped onto its first bin -- the same approach as
// MuonSFYear::pt_min. Revisit if EGM adds a low-pt reco SF for 2025.
struct ElectronRecoLowPt {
    std::string working_point;
    float pt_min; // minimum valid pt, verified against JSON bin edges
};
const std::unordered_map<std::string, ElectronRecoLowPt> electronRecoLowPt = {
    {"2016preVFP",            {"RecoBelow20", 10.1}},
    {"2016postVFP",           {"RecoBelow20", 10.1}},
    {"2017",                  {"RecoBelow20", 10.1}},
    {"2018",                  {"RecoBelow20", 10.1}},
    {"2022Re-recoBCD",        {"RecoBelow20", 10.1}},
    {"2022Re-recoE+PromptFG", {"RecoBelow20", 10.1}},
    {"2023PromptC",           {"RecoBelow20", 10.1}},
    {"2023PromptD",           {"RecoBelow20", 10.1}},
    {"2024Prompt",            {"RecoBelow20", 10.1}},
    {"2025",                  {"Reco20to75",  20.1}} // 2025 json has no RecoBelow20
};

struct ElectronIDConfig {
    std::unordered_map<std::string, std::string> correction_name_map;
    std::string working_point; // Loose or Tight
};

const ElectronIDConfig electronID_loose = {
    {
        {"2016preVFP",            "UL-Electron-ID-SF"},
        {"2016postVFP",           "UL-Electron-ID-SF"},
        {"2017",                  "UL-Electron-ID-SF"},
        {"2018",                  "UL-Electron-ID-SF"},
        {"2022Re-recoBCD",        "Electron-ID-SF"},
        {"2022Re-recoE+PromptFG", "Electron-ID-SF"},
        {"2023PromptC",           "Electron-ID-SF"},
        {"2023PromptD",           "Electron-ID-SF"},
        {"2024Prompt",            "Electron-ID-SF"},
        {"2025",                  "Electron-ID-SF"}
    },
    "Loose"
};

const ElectronIDConfig electronID_tight = {
    {
        {"2016preVFP",            "UL-Electron-ID-SF"},
        {"2016postVFP",           "UL-Electron-ID-SF"},
        {"2017",                  "UL-Electron-ID-SF"},
        {"2018",                  "UL-Electron-ID-SF"},
        {"2022Re-recoBCD",        "Electron-ID-SF"},
        {"2022Re-recoE+PromptFG", "Electron-ID-SF"},
        {"2023PromptC",           "Electron-ID-SF"},
        {"2023PromptD",           "Electron-ID-SF"},
        {"2024Prompt",            "Electron-ID-SF"},
        {"2025",                  "Electron-ID-SF"}
    },
    "Tight"
};

const std::unordered_map<std::string, std::vector<std::string>> electronWorkingPointSFs = {
    {"_weight_electron_reco_looseid", {
        "_weight_electron_reco",
        "_weight_electron_id_loose"
    }},
    {"_weight_electron_reco_tightid", {
        "_weight_electron_reco",
        "_weight_electron_id_tight"
    }}
};

// Keep trigger SF machinery available for 1lep
const std::unordered_map<std::string, correction::CorrectionSet> electronTriggerScaleFactors = {
    {"2022Re-recoBCD",        *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22CDSep23-Summer22-NanoAODv12/2025-12-15/electronHlt.json.gz")},
    {"2022Re-recoE+PromptFG", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-12-15/electronHlt.json.gz")},
    {"2023PromptC",           *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23CSep23-Summer23-NanoAODv12/2025-12-15/electronHlt.json.gz")},
    {"2023PromptD",           *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-12-15/electronHlt.json.gz")}
};

const std::unordered_map<std::string, std::string> electronTriggerScaleFactors_yearmap = {
    {"2022Re-recoBCD",        "Electron-HLT-SF"},
    {"2022Re-recoE+PromptFG", "Electron-HLT-SF"},
    {"2023PromptC",           "Electron-HLT-SF"},
    {"2023PromptD",           "Electron-HLT-SF"}
};

// New reco+ID WP machinery
RNode applyElectronRecoScaleFactors(std::unordered_map<std::string, correction::CorrectionSet> cset_electron, RNode df, std::string output_name = "_weight_electron_reco");
RNode applyElectronIDScaleFactors(std::unordered_map<std::string, correction::CorrectionSet> cset_electron, ElectronIDConfig config, std::string output_name, RNode df);
RNode combineElectronScaleFactorWeightsByKey(RNode df, std::string output_name, std::vector<std::string> input_keys);
RNode applyElectronWorkingPointSFs(RNode df, bool isData, std::vector<std::string> wp_keys);

// Keep trigger SF declaration available
RNode applyElectronTriggerScaleFactors(std::unordered_map<std::string, correction::CorrectionSet> cset_electron, std::unordered_map<std::string, std::string> year_map, RNode df);


/*
############################################
B-TAGGING SFs
############################################
*/
const std::unordered_map<std::string, correction::CorrectionSet> bTaggingScaleFactors = {
    {"2016preVFP", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2016preVFP-UL-NanoAODv15/latest/btagging.json.gz")},
    {"2016postVFP", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2016postVFP-UL-NanoAODv15/latest/btagging.json.gz")},
    {"2017", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2017-UL-NanoAODv15/latest/btagging.json.gz")},
    {"2018", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2018-UL-NanoAODv15/latest/btagging.json.gz")},
    {"2022Re-recoBCD", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-22CDSep23-Summer22-NanoAODv12/latest/btagging.json.gz")},
    {"2022Re-recoE+PromptFG", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-22EFGSep23-Summer22EE-NanoAODv12/latest/btagging.json.gz")},
    {"2023PromptC", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-23CSep23-Summer23-NanoAODv12/latest/btagging.json.gz")},
    {"2023PromptD", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-23DSep23-Summer23BPix-NanoAODv12/latest/btagging.json.gz")},
    {"2024Prompt", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/btagging.json.gz")},
    {"2025", *CorrectionSet::from_file("/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-25Prompt-Summer24-NanoAODv15/latest/btagging.json.gz")}
};

// Deliberately lazy: --btag_eff must be able to create this payload when it
// does not yet exist in a fresh checkout or worker sandbox.
correction::CorrectionSet loadBTagEfficiencyCorrectionSet(const std::string &year);

const std::unordered_map<std::string, std::string> bTaggingScaleFactors_HF_corrname = {
    {"2016preVFP", "UParTAK4_comb"},
    {"2016postVFP", "UParTAK4_comb"},
    {"2017", "UParTAK4_comb"},
    {"2018", "UParTAK4_comb"},
    {"2024Prompt", "UParTAK4_comb"},
    {"2025", "UParTAK4_comb"}
};

const std::unordered_map<std::string, std::string> bTaggingScaleFactors_LF_corrname = {
    {"2016preVFP", "UParTAK4_light"},
    {"2016postVFP", "UParTAK4_light"},
    {"2017", "UParTAK4_light"},
    {"2018", "UParTAK4_light"},
    {"2024Prompt", "UParTAK4_light"},
    {"2025", "UParTAK4_light"}
};

using BTagCorrectionRef = std::decay_t<decltype(std::declval<const correction::CorrectionSet &>().at(std::declval<std::string>()))>;

struct BTagEfficiencyContext {
    std::string efficiency_sample;
    std::string efficiency_name;
    std::shared_ptr<correction::CorrectionSet> efficiency_set;
    BTagCorrectionRef efficiency;
    BTagCorrectionRef hf_sf;
    BTagCorrectionRef lf_sf;
};

using BTagEfficiencyContexts = std::map<std::pair<std::string, std::string>, BTagEfficiencyContext>;

BTagEfficiencyContexts prepareBTagEfficiencyContexts(
    const std::vector<BTagEfficiencyMetadata> &metadata,
    const std::string &channel,
    const std::vector<std::string> &working_points);

RNode applyBTaggingScaleFactors(const std::string &channel,
                                const std::vector<std::string> &working_points,
                                const BTagEfficiencyContexts &contexts, RNode df);
void resetBTagDiagnostics();
void printBTagDiagnostics(std::ostream &out = std::cout);

/*
############################################
OTHER SFs
############################################
*/

RNode applyEWKCorrections(CorrectionSet cset_ewk, RNode df);
const auto cset_ewk = *CorrectionSet::from_file("corrections/scalefactors/ewk/EWK.json");

RNode applyL1PreFiringReweighting(RNode df);
RNode applyPSWeight_FSR(RNode df);
RNode applyPSWeight_ISR(RNode df);
RNode applyLHEScaleWeight_muF(RNode df);
RNode applyLHEScaleWeight_muR(RNode df);
RNode applyLHEWeights_pdf(RNode df);

RNode applyDataWeights(RNode df);
RNode applyMCWeights(RNode df, const std::string &channel, bool apply_btag_sf,
                     const std::vector<std::string> &btag_working_points,
                     const BTagEfficiencyContexts &btag_contexts);

#endif //WEIGHTS
