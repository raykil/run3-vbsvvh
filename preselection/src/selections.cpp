#include "selections.h"
#include "cutflow.h"
#include "weights.h"

namespace {
// AK4 good-jet selection as a string template, parametrised by the pt-column name.
// Used for both the nominal mask (Jet_pt) and the per-source JES variations
// (Jet_pt_jesAbsoluteUp, ...). Must be combined with `&& !Jet_vetoMap` after the veto
// map has been applied (the nominal path does this via Redefine; the variation loop
// appends it explicitly).
inline std::string ak4GoodJetSelectionExpr(const std::string& ptCol) {
    return "_dR_ak4_lep > 0.4 &&"
           "((isRun3 && ((" + ptCol + " > 20 && (abs(Jet_eta) <= 2.5 || abs(Jet_eta) >= 3.0) && abs(Jet_eta) < 5.0) || "
                        "(" + ptCol + " > 50 && abs(Jet_eta) > 2.5 && abs(Jet_eta) < 3.0))) ||"
           "(isRun2 && " + ptCol + " > 20 && abs(Jet_eta) < 5.0)) &&  "
           "(Jet_jetId >= 2)";
}

inline std::string ak8GoodJetSelectionExpr(const std::string& ptCol) {
    return "_dR_ak8_lep > 0.8 && "
           + ptCol + " > 250 && "
           "abs(FatJet_eta) <= 2.5 && "
           "FatJet_msoftdrop > 40 && "
           "FatJet_jetId > 0";
}

// Kinematic-variation suffixes (JES + JER) whose varied jet columns actually exist in
// this graph. Presence-checked rather than assumed: variations are MC-only (and only
// with --systs given), so on data or in the default nominal-only mode this returns empty and the
// selection stays nominal-only. Deriving the set from the columns the correction path
// actually defined means there is no flag to keep in sync with that path.
inline std::vector<std::string> activeKinVariations(RNode df) {
    std::vector<std::string> out;
    for (const auto& sfx : kinematicVariationSuffixes()) {
        if (df.HasColumn("Jet_pt_" + sfx) && df.HasColumn("FatJet_pt_" + sfx))
            out.push_back(sfx);
    }
    return out;
}

// Per-variation channel-pass-flag generation. Each channel filter that depends on a
// jet-multiplicity cut emits a boolean per event for each variation v ∈ {nom} ∪
// activeKinVariations() - passes_<channel>_<v> - then the channel filter is OR over
// those flags. This keeps events that pass the channel under ANY variation, which is
// the only way a single output file can carry per-variation histograms downstream.
//
// `expr_for(sfx)` returns the full pass condition for variation `sfx`. sfx == "" means
// nominal; the lambda should return the nominal condition string in that case.
template <typename F>
RNode definePerVariationPassFlags(RNode df, const std::string& channel, F&& expr_for) {
    df = df.Define("passes_" + channel + "_nom", expr_for(std::string{}));
    for (const auto& sfx : activeKinVariations(df)) {
        df = df.Define("passes_" + channel + "_" + sfx, expr_for(sfx));
    }
    return df;
}

inline std::string orPassExpr(RNode df, const std::string& channel) {
    std::string out = "passes_" + channel + "_nom";
    for (const auto& sfx : activeKinVariations(df)) {
        out += " || passes_" + channel + "_" + sfx;
    }
    return out;
}
} // anonymous namespace

// MET filters
// https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2
RNode METFilters(RNode df_) {
    return df_.Filter("(isRun3 && (Flag_goodVertices && "
            "Flag_globalSuperTightHalo2016Filter && "
            "Flag_EcalDeadCellTriggerPrimitiveFilter && "
            "Flag_BadPFMuonFilter && "
            "Flag_BadPFMuonDzFilter && "
            "Flag_hfNoisyHitsFilter &&"
            "Flag_eeBadScFilter && "
            "Flag_ecalBadCalibFilter)) || "
        "(isRun2 && (Flag_goodVertices && "
            "Flag_globalSuperTightHalo2016Filter && "
            "Flag_HBHENoiseFilter && "
            "Flag_HBHENoiseIsoFilter && "
            "Flag_EcalDeadCellTriggerPrimitiveFilter && "
            "Flag_BadPFMuonFilter &&"
            "Flag_BadPFMuonDzFilter && "
            "Flag_eeBadScFilter && "
            "(!(is2017 || is2018) || Flag_ecalBadCalibFilter)))"
        );
}

RNode TriggerSelections(RNode df_, std::string trigger_logic_string) {

    // Add default values to the df for all triggers that show up in the trigger_logic_string
    std::string trigger_condition = trigger_logic_string;
    std::regex hlt_regex("HLT_[a-zA-Z0-9_]+");
    auto hlt_begin = std::sregex_iterator(trigger_condition.begin(), trigger_condition.end(), hlt_regex);
    auto hlt_end = std::sregex_iterator();
    std::vector<std::string> seen_triggers;

    for (auto it = hlt_begin; it != hlt_end; ++it)
    {
        std::string hlt_name = it->str(0);
        if (std::find(seen_triggers.begin(), seen_triggers.end(), hlt_name) != seen_triggers.end())
            continue;
        seen_triggers.push_back(hlt_name);
        df_ = df_.DefaultValueFor(hlt_name, (bool)false);
    }

    // Filter based on this the trigger logic
    return df_.Filter(trigger_condition, "C1: Trigger Selection");
}

// Same defaulting TriggerSelections does, but with no Filter: the decision is
// stored for offline use rather than applied here.
RNode storeTriggerBranches(RNode df_, const std::vector<std::string> &paths)
{
    for (const auto &path : paths) df_ = df_.DefaultValueFor(path, (bool)false);
    return df_;
}


// Ele selection
RNode ElectronSelections(RNode df_)
{
    auto df = df_.Define("Electron_SC_eta", "Electron_eta + Electron_deltaEtaSC");

    // Veto selection
    df = df.Define(
        "_vetoElectrons",
        "Electron_pt > 10 &&"
        "abs(Electron_SC_eta) < 2.5 && "
        "((abs(Electron_SC_eta) <= 1.479 && abs(Electron_dxy) <= 0.05 && abs(Electron_dz) < 0.1) || ((abs(Electron_SC_eta) > 1.479) && abs(Electron_dxy) <= 0.1 && abs(Electron_dz) < 0.2)) && "
        "Electron_cutBased >= 1"
    );

    // Loose selection
    df = df.Define(
        "_looseElectrons",
        "_vetoElectrons &&"
        "Electron_cutBased >= 2"
    );

    // We will write out the electron object
    df = applyObjectMaskNewAffix(df, "_vetoElectrons", "Electron", "electron");

    // Append the masks to electron object
    df = df.Define("electron_isLoose",  "electron_cutBased >=2");
    df = df.Define("electron_isMedium", "electron_cutBased >=3");
    df = df.Define("electron_isTight",  "electron_cutBased >=4");

    // Define the counts of each
    df = df.Define("nElectron_Veto", "nElectron == 0 ? 0 : Sum(_vetoElectrons)");
    df = df.Define("nElectron_Loose", "Sum(electron_isLoose)");
    df = df.Define("nElectron_Tight", "Sum(electron_isTight)");

    return df;
}

// Muon selections
RNode MuonSelections(RNode df_)
{

    // Loose selection
    auto df = df_.Define(
        "_looseMuons",
        "Muon_pt > 10 && "
        "Muon_pfIsoId >= 2 && "
        "abs(Muon_eta) < 2.4 && "
        "abs(Muon_dxy) < 0.2 && "
        "abs(Muon_dz) < 0.5 && "
        "abs(Muon_sip3d) < 8 && "
        "Muon_looseId"
    );

    // Medium selection
    df = df.Define(
        "_mediumMuons",
        "_looseMuons && "
        "Muon_pfIsoId >= 3 && "
        "Muon_mediumId"
    );

    // We will write out the muon object
    df = applyObjectMaskNewAffix(df, "_looseMuons", "Muon", "muon");

    // Append the masks to electron object
    df = df.Define("muon_isMedium",    "muon_mediumId && (muon_pfIsoId >=3)");
    df = df.Define("muon_isTight",     "muon_mediumId && (muon_pfIsoId >=4)");
    df = df.Define("muon_isVeryTight", "muon_mediumId && (muon_pfIsoId >=5)");

    // Define the counts of each
    df = df.Define("nMuon_Loose", "nMuon == 0 ? 0 : Sum(_looseMuons)");
    df = df.Define("nMuon_Medium", "Sum(muon_isMedium)");
    df = df.Define("nMuon_Tight", "Sum(muon_isTight)");

    return df;
}

// Put ele and mu together into lepton collection
RNode LeptonSelections(RNode df_)
{
    auto df = ElectronSelections(df_);
    df = MuonSelections(df);

    // Counting lepton collections: used for SF calculation and channel counting
    //     - These are the leptons that define the channel (loose WP for both flavors)
    //     - Prefixed with _ so they are not written to the output snapshot
    //     - Muons: loose muons (muon_* collection is already loose)
    //     - Electrons: loose electrons (i.e., loose subset of the veto electron collection)
    df = df.Define("_muonSel_pt",  "muon_pt")
           .Define("_muonSel_eta", "muon_eta");
    df = df.Define("_electronSel_pt",     "electron_pt[electron_isLoose]")
           .Define("_electronSel_eta",    "electron_eta[electron_isLoose]")
           .Define("_electronSel_SC_eta", "electron_SC_eta[electron_isLoose]");
    // Counter for these selection level leptons
    df = df.Define("nLep_Sel", "nMuon_Loose + nElectron_Loose");

    return df.Define("lepton_pt", "Concatenate(electron_pt, muon_pt)")
            .Define("_leptonSorted", "Argsort(-lepton_pt)")
            .Redefine("lepton_pt", "Take(lepton_pt, _leptonSorted)")
            .Define("lepton_eta", "Take(Concatenate(electron_eta, muon_eta), _leptonSorted)")
            .Define("lepton_phi", "Take(Concatenate(electron_phi, muon_phi), _leptonSorted)")
            .Define("lepton_mass", "Take(Concatenate(electron_mass, muon_mass), _leptonSorted)")
            .Define("lepton_charge", "Take(Concatenate(electron_charge, muon_charge), _leptonSorted)");
}


// Small-radius AK4 jet selection
RNode AK4JetsSelection(RNode df_, bool cleanAgainstFJ, std::string affix)
{
    auto df = DefineOrRedefine(df_, "_dR_ak4_lep", VVdR, {"Jet_eta", "Jet_phi", "lepton_eta", "lepton_phi"});

    // Optionally define the dR between AK4 jets and fat jets, and include it in the AK4 selection if cleanAgainstFJ is set
    std::string fj_cut = "";
    if (cleanAgainstFJ) {
        df = df.Define("_dR_ak4_fatjet", VVdR, {"Jet_eta", "Jet_phi", "fatjet_eta", "fatjet_phi"});
        fj_cut = "_dR_ak4_fatjet > 0.8 && ";
    }

    df = DefineOrRedefine(df, "_good_ak4jets", "_dR_ak4_lep > 0.4 && " + fj_cut +
                                    "((isRun3 && ((Jet_pt > 20 && (abs(Jet_eta) <= 2.5 || abs(Jet_eta) >= 3.0) && abs(Jet_eta) < 5.0) || "
                                    "(Jet_pt > 50 && abs(Jet_eta) > 2.5 && abs(Jet_eta) < 3.0))) ||"
                                    "(isRun2 && Jet_pt > 20 && abs(Jet_eta) < 5.0)) &&  "
                                    "(Jet_jetId >= 2)"); // NanoAOD jetID convention https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD#Jets
                                                        // should still work for skims < v28, which set jetId==3 : "pass tight ID, fail tightLepVeto", jetId==7 : "pass tight and tightLepVeto ID"

    // Jet veto maps. Run 3 drops the whole event (the veto exists to remove the spurious
    // MET a bad-region jet induces, which masking the jet alone would not undo); Run 2
    // drops only the jets. Both are skipped under --no_jetveto, which leaves Jet_vetoMap
    // defined but unapplied -- DEBUGGING ONLY, see corrections.h.
    const std::string vetoTerm = applyJetVetoMapsEnabled() ? " && !Jet_vetoMap" : "";
    if (applyJetVetoMapsEnabled()) {
        df = df.Filter("(isRun2) || (isRun3 && !Any(Jet_vetoMap))");
        df = df.Redefine("_good_ak4jets", "_good_ak4jets && !Jet_vetoMap");
    }

    df = applyObjectMaskNewAffix(df, "_good_ak4jets", "Jet", affix);
    df = df.Define("ht_" + affix, "Sum(" + affix + "_pt)");

    // Nominal + per-variation good-jet masks stored at the NanoAOD (capital-collection) level.
    // Defined once, alongside the primary FJ-cleaned "jet" collection, and AFTER
    // applyObjectMaskNewAffix so they are not aliased into lowercase jet_isGood columns.
    // Downstream coffea uses Jet_isGood (nominal) and Jet_isGood_<sfx> (variations)
    // uniformly to construct jet collections from the shared Jet_* arrays.
    //
    // Each per-variation mask is fat-jet-cleaned against the good fat jets FOR THAT
    // variation (FatJet_isGood_<sfx>), so Jet_isGood_<sfx> parallels the FJ-cleaned nominal
    // Jet_isGood (= _good_ak4jets for the "jet" affix). Only the good-fatjet membership
    // changes with the variation; the ak4-ak8 dR geometry itself is JEC-invariant (eta/phi
    // are unchanged by JEC). VVdR returns 999 for every AK4 jet when a variation has no good
    // fat jets, so the >0.8 cut then passes (no cleaning) -- matching the nominal fj_cut.
    //
    // njet_<sfx> = Sum(Jet_isGood_<sfx>) then parallels the nominal `njet` (defined inside
    // applyObjectMaskNewAffix, also FJ-cleaned) and feeds the per-variation channel-pass flags.
    if (affix == "jet") {
        df = df.Define("Jet_isGood", "_good_ak4jets");
        for (const auto& sfx : activeKinVariations(df)) {
            df = df.Define("_fatjet_eta_" + sfx, "FatJet_eta[FatJet_isGood_" + sfx + "]");
            df = df.Define("_fatjet_phi_" + sfx, "FatJet_phi[FatJet_isGood_" + sfx + "]");
            df = df.Define("_dR_ak4_fatjet_" + sfx, VVdR,
                           {"Jet_eta", "Jet_phi", "_fatjet_eta_" + sfx, "_fatjet_phi_" + sfx});
            df = df.Define("Jet_isGood_" + sfx,
                           ak4GoodJetSelectionExpr("Jet_pt_" + sfx)
                               + vetoTerm + " && _dR_ak4_fatjet_" + sfx + " > 0.8");
            df = df.Define("njet_" + sfx, "Sum(Jet_isGood_" + sfx + ")");
        }
    }
    // Nominal + per-variation masks for the non-FJ-cleaned collection, used by
    // VBSTagging in channels that tag on "jetNoFJClean" (1lep_2FJ). No FJ cleaning to
    // re-evaluate, so the mask is just the pt-varied good-jet expression + veto map.
    // The Jet_ prefix means the masks are written to the output (snapshot keeps all
    // Jet_* columns), mirroring Jet_isGood / Jet_isGood_<sfx> for the cleaned set.
    else if (affix == "jetNoFJClean") {
        df = df.Define("Jet_isGoodNoFJClean", "_good_ak4jets");
        for (const auto& sfx : activeKinVariations(df)) {
            df = df.Define("Jet_isGoodNoFJClean_" + sfx,
                           ak4GoodJetSelectionExpr("Jet_pt_" + sfx) + vetoTerm);
            df = df.Define("njetNoFJClean_" + sfx, "Sum(Jet_isGoodNoFJClean_" + sfx + ")");
        }
    }
    return df;
}

// Define per-jet properties (b-tagging, veto maps) independent of jet selection
RNode AK4JetProperties(RNode df_)
{
    df_ = applyJetVetoMaps(df_);
    return df_.Define("Jet_isTightBTag", isbTagTight, {"year", "Jet_btagUParTAK4B"})
              .Define("Jet_isMediumBTag", isbTagMedium, {"year", "Jet_btagUParTAK4B"})
              .Define("Jet_isLooseBTag", isbTagLoose, {"year", "Jet_btagUParTAK4B"})
              .Define("Jet_isExtraTightBTag", isbTagExtraTight, {"year", "Jet_btagUParTAK4B"})
              .Define("Jet_isExtraExtraTightBTag", isbTagExtraExtraTight, {"year", "Jet_btagUParTAK4B"});
}


// Fat jet AK8 selection
RNode AK8JetsSelection(RNode df_)
{
    auto df = df_.Define("_dR_ak8_lep", VVdR, {"FatJet_eta", "FatJet_phi", "lepton_eta", "lepton_phi"})
                  .Define("_good_ak8jets", ak8GoodJetSelectionExpr("FatJet_pt"));

    // GloParT regressed mass: the calibrated AK8 mass used by the analysis. massCorrX2p is a
    // ratio predicting the particle-level mass from the RAW jet, so it multiplies the raw mass
    // snapshotted before JEC (_FatJet_rawmass, from snapshotRawJetKinematics in corrections.cpp)
    // rather than the JEC/JER-corrected FatJet_mass. It replaces the JEC/JER mass calibration
    // for AK8 and is not stacked on top of it. Defined before applyObjectMaskNewAffix so the
    // masked fatjet_massGloParT3 alias is created with the rest of the collection.
    df = df.Define("FatJet_massGloParT3", "FatJet_globalParT3_massCorrX2p * _FatJet_rawmass");

    // GloParT tagging discriminants, built from the raw score outputs as signal / (signal + QCD):
    //   Hbb = Xbb / (Xbb + QCD)
    //   Wqq = (Xqq/3 + Xcs) / (Xqq/3 + Xcs + QCD)
    // Unlike the mass these are kinematics-independent, so they need no raw-quantity snapshot.
    // Jets with a vanishing denominator (no score at all) get -1 rather than a NaN.
    auto vsQCD = [](const RVec<float>& sig, const RVec<float>& qcd) {
        RVec<float> out(sig.size(), -1.0f);
        for (size_t i = 0; i < sig.size(); ++i) {
            const float den = sig[i] + qcd[i];
            if (den > 0.0f) out[i] = sig[i] / den;
        }
        return out;
    };
    df = df.Define("_FatJet_Vqq_score", "FatJet_globalParT3_Xqq / 3.0f + FatJet_globalParT3_Xcs")
           .Define("FatJet_Hbb", vsQCD, {"FatJet_globalParT3_Xbb", "FatJet_globalParT3_QCD"})
           .Define("FatJet_Wqq", vsQCD, {"_FatJet_Vqq_score",     "FatJet_globalParT3_QCD"});

    df = applyObjectMaskNewAffix(df, "_good_ak8jets", "FatJet", "fatjet");
    df = df.Define("ht_fatjets", "Sum(fatjet_pt)");

    // Nominal good-fat-jet mask - defined AFTER applyObjectMaskNewAffix so it is not
    // aliased into fatjet_isGood. FatJet_isGood (nominal) mirrors the per-variation
    // FatJet_isGood_<sfx> branches; coffea uses them uniformly. The nominal count is
    // already available as "nfatjet" (from applyObjectMaskNewAffix above), matching the
    // AK4 convention (njet / njet_<sfx>), so no separate nominal count is defined here.
    df = df.Define("FatJet_isGood", "_good_ak8jets");

    // Per-variation good-fatjet masks + counts. JES/JER affect FatJet_pt only;
    // FatJet_msoftdrop is unchanged. Defined AFTER applyObjectMaskNewAffix for the same
    // reason as the AK4 case. Named nfatjet_<sfx> to parallel the nominal "nfatjet" and
    // AK4's njet / njet_<sfx>.
    for (const auto& sfx : activeKinVariations(df)) {
        df = df.Define("FatJet_isGood_" + sfx, ak8GoodJetSelectionExpr("FatJet_pt_" + sfx));
        df = df.Define("nfatjet_" + sfx, "Sum(FatJet_isGood_" + sfx + ")");
    }
    return df;
}

// Perform VBS jet tagging via VBSBDTInfer and compute VBS pair kinematics
// The nominal or a variation with < 2 taggable jets gets sentinel values instead (indices -1, score/kinematics -999).
RNode VBSTagging(RNode df_, std::string jetCollectionName = "jet")
{
    // All-jet-level mask stem for the tagging collection. The per-variation masks are
    // defined in AK4JetsSelection for both stems.
    const bool fjCleaned = (jetCollectionName == "jet");
    const std::string maskStem = fjCleaned ? "Jet_isGood" : "Jet_isGoodNoFJClean";

    // Continue to store the full four vector of the two VBS candidates when running
    // on nominal for backwards compatibility.
    auto df = df_
        .Define("_vbs_candidate_jet_pairs", VBSBDTInfer, {jetCollectionName + "_pt", jetCollectionName + "_eta", jetCollectionName + "_phi", jetCollectionName + "_mass", "isRun2"})
        .Define("vbs_jet1_idx", "static_cast<int>(_vbs_candidate_jet_pairs[0])") // indices in the "good jet" collection
        .Define("vbs_jet2_idx", "static_cast<int>(_vbs_candidate_jet_pairs[1])")
        .Define("vbs_score", "vbs_jet1_idx >= 0 ? _vbs_candidate_jet_pairs[2] : -999.f")
        .Define("vbs_jet1_pt",   "vbs_jet1_idx >= 0 ? " + jetCollectionName + "_pt[vbs_jet1_idx]   : -999.f")
        .Define("vbs_jet1_eta",  "vbs_jet1_idx >= 0 ? " + jetCollectionName + "_eta[vbs_jet1_idx]  : -999.f")
        .Define("vbs_jet1_phi",  "vbs_jet1_idx >= 0 ? " + jetCollectionName + "_phi[vbs_jet1_idx]  : -999.f")
        .Define("vbs_jet1_mass", "vbs_jet1_idx >= 0 ? " + jetCollectionName + "_mass[vbs_jet1_idx] : -999.f")
        .Define("vbs_jet2_pt",   "vbs_jet2_idx >= 0 ? " + jetCollectionName + "_pt[vbs_jet2_idx]   : -999.f")
        .Define("vbs_jet2_eta",  "vbs_jet2_idx >= 0 ? " + jetCollectionName + "_eta[vbs_jet2_idx]  : -999.f")
        .Define("vbs_jet2_phi",  "vbs_jet2_idx >= 0 ? " + jetCollectionName + "_phi[vbs_jet2_idx]  : -999.f")
        .Define("vbs_jet2_mass", "vbs_jet2_idx >= 0 ? " + jetCollectionName + "_mass[vbs_jet2_idx] : -999.f")
        .Define("vbs_mjj",    "vbs_jet1_idx >= 0 ? (ROOT::Math::PtEtaPhiMVector(vbs_jet1_pt, vbs_jet1_eta, vbs_jet1_phi, vbs_jet1_mass) + "
                              "ROOT::Math::PtEtaPhiMVector(vbs_jet2_pt, vbs_jet2_eta, vbs_jet2_phi, vbs_jet2_mass)).M() : -999.")
        .Define("vbs_detajj", "vbs_jet1_idx >= 0 ? abs(vbs_jet1_eta - vbs_jet2_eta) : -999.");

    // Nominal VBS-jet indices in the all-jet (NanoAOD Jet_*) collection, corresponding
    // to the same physical jets as vbs_jet1/2_idx; stored so downstream
    // handles nominal and variations uniformly via Jet_*[vbs_jet*_Jetidx*].
    df = df.Define("vbs_jet1_Jetidx", "vbs_jet1_idx >= 0 ? static_cast<int>(ROOT::VecOps::Nonzero(" + maskStem + ")[vbs_jet1_idx]) : -1")
           .Define("vbs_jet2_Jetidx", "vbs_jet2_idx >= 0 ? static_cast<int>(ROOT::VecOps::Nonzero(" + maskStem + ")[vbs_jet2_idx]) : -1");

    // Per-variation VBS tagging: re-run the BDT on the variation's good-jet set with its
    // varied pt/mass (eta/phi are JEC-invariant). Only the two all-jet-frame indices and
    // the score are stored per variation - downstream can rebuild the kinematics from
    // Jet_*[vbs_jet*_Jetidx*], keeping the branch count small.
    for (const auto& sfx : activeKinVariations(df)) {
        df = df.Define("_vbs_pair_" + sfx, VBSBDTInferMasked,
                       {maskStem + "_" + sfx, "Jet_pt_" + sfx, "Jet_eta", "Jet_phi", "Jet_mass_" + sfx, "isRun2"});
        df = df.Define("vbs_jet1_Jetidx_" + sfx, "static_cast<int>(_vbs_pair_" + sfx + "[0])");
        df = df.Define("vbs_jet2_Jetidx_" + sfx, "static_cast<int>(_vbs_pair_" + sfx + "[1])");
        df = df.Define("vbs_score_" + sfx, "_vbs_pair_" + sfx + "[2]");
    }
    return df;
}



///////////////// Main channel selection block /////////////////
RNode runPreselection(RNode df_, std::string channel, bool noCut, bool isData)
{

    Cutflow::Add(df_, "All events");

    // Standard MET filters
    auto df = METFilters(df_);
    Cutflow::Add(df, "C0: MET filters");

    // Perform lepton selection first (these have highest priority)
    df = LeptonSelections(df);

    // Fat jet selection (cleaned against leptons)
    df = AK8JetsSelection(df);

    // Do the ak4 jet selection
    df = AK4JetProperties(df);
    df = AK4JetsSelection(df, /*cleanAgainstFJ=*/true, "jet");
    df = AK4JetsSelection(df, /*cleanAgainstFJ=*/false, "jetNoFJClean");

    // Get ahold of the relevant lepton SFs (will choose which ones to actually apply in each channel)
    df = applyMuonWorkingPointSFs(df, isData, {
        "_weight_muon_looseid_looseiso",
        "_weight_muon_mediumid_tightiso",
        "_weight_muon_tightid_tightiso"
    });
    df = applyElectronWorkingPointSFs(df, isData, {
        "_weight_electron_reco_looseid",
        "_weight_electron_reco_tightid"
    });


    // Passthrough
    if (noCut) return df;
    if (channel == "all_events"){
        df = df.Filter(
            "nMuon_Loose > -1", // Probably there is a better way to write a pass through
            "C2: all_events"
        );
    }

    ////////// Each channel selections //////////

    // 0lep_0FJ
    else if (channel == "0lep_0FJ"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_0lep0FJ);
        Cutflow::Add(df, "C1: Trigger selection");

        // Channel orthogonality selection
        df = definePerVariationPassFlags(df, "0lep_0FJ", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            return "(nLep_Sel == 0) && (" + fjCountCol + " == 0)";
        });
        df = df.Filter(orPassExpr(df, "0lep_0FJ"), "C2: 0lep_0FJ");

        df = df.Define("met_significance", "PuppiMET_significance")
                .Define("met_uncorrPt", "PuppiMET_pt")
                .Define("met_uncorrPhi", "PuppiMET_phi");
    }

    // 0lep_1FJ
    else if (channel == "0lep_1FJ"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_0lep1FJ);
        Cutflow::Add(df, "C1: Trigger selection");

        // Channel orthogonality selection
        df = definePerVariationPassFlags(df, "0lep_1FJ", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            return "(nLep_Sel == 0) && (" + fjCountCol + " == 1)";
        });
        df = df.Filter(orPassExpr(df, "0lep_1FJ"), "C2: 0lep_1FJ");
    }

    // 0lep_1FJ_met
    else if (channel == "0lep_1FJ_met"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_met);
        // The trigger choice is currently applied offline instead.
        // These are the triggers whose decisions are written out.
        const std::vector<std::string> met_trigger_paths = {
            "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
            "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight",
            "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
            "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
        };
        df = storeTriggerBranches(df, met_trigger_paths);
        Cutflow::Add(df, "C1: Trigger selection");

        // Channel orthogonality selection
        df = definePerVariationPassFlags(df, "0lep_1FJ_met", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            return "(nLep_Sel == 0) && (" + fjCountCol + " == 1)";
        });
        df = df.Filter(orPassExpr(df, "0lep_1FJ_met"), "C2: 0lep_1FJ");

        df = df.DefaultValueFor("Pileup_nTrueInt", (float)-1.f)
                .Define("met_significance", "PuppiMET_significance")
                .Define("met_uncorrPt", "PuppiMET_pt")
                .Define("met_uncorrPhi", "PuppiMET_phi")
                .Define("pileup_nTrueInt", "Pileup_nTrueInt")
                .Define("pv_npvsGood", "PV_npvsGood");
    }

    // 0lep_2FJ
    else if (channel == "0lep_2FJ"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_ht);
        Cutflow::Add(df, "C1: Trigger selection");

        // Channel orthogonality selection
        df = definePerVariationPassFlags(df, "0lep_2FJ", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            return "(nLep_Sel == 0) && (" + fjCountCol + " == 2)";
        });
        df = df.Filter(orPassExpr(df, "0lep_2FJ"), "C2: 0lep_2FJ");
    }

    // 0lep_2FJ_met
    else if (channel == "0lep_2FJ_met"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_met);
        // The trigger choice is currently applied offline instead.
        // These are the triggers whose decisions are written out.
        const std::vector<std::string> met_trigger_paths = {
            "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
            "HLT_PFMETNoMu110_PFMHTNoMu110_IDTight",
            "HLT_PFMETNoMu140_PFMHTNoMu140_IDTight",
            "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
        };
        df = storeTriggerBranches(df, met_trigger_paths);
        Cutflow::Add(df, "C1: Trigger selection");

        // Channel orthogonality selection
        df = definePerVariationPassFlags(df, "0lep_2FJ_met", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            return "(nLep_Sel == 0) && (" + fjCountCol + " == 2)";
        });
        df = df.Filter(orPassExpr(df, "0lep_2FJ_met"), "C2: 0lep_2FJ");

        df = df.DefaultValueFor("Pileup_nTrueInt", (float)-1.f)
                .Define("met_significance", "PuppiMET_significance")
                .Define("met_uncorrPt", "PuppiMET_pt")
                .Define("met_uncorrPhi", "PuppiMET_phi")
                .Define("pileup_nTrueInt", "Pileup_nTrueInt")
                .Define("pv_npvsGood", "PV_npvsGood");
    }

    // 0lep_3FJ
    else if (channel == "0lep_3FJ"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_ht);
        Cutflow::Add(df, "C1: Trigger selection");

        // Channel orthogonality selection
        df = definePerVariationPassFlags(df, "0lep_3FJ", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            return "(nLep_Sel == 0) && (" + fjCountCol + " == 3)";
        });
        df = df.Filter(orPassExpr(df, "0lep_3FJ"), "C2: 0lep_3FJ");
    }

    // 1lep_1FJ - fatjet + njet cuts must combine inside a single per-variation pass flag,
    // because OR-ing over two consecutive jet-count filters would mix variations across
    // different cuts. Cutflow C3 + C4 (separate fatjet/njet steps) collapse into a single
    // "any-variation jet selection" entry.
    else if (channel == "1lep_1FJ"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_singlelep);
        Cutflow::Add(df, "C1: Trigger selection");

        df = lepSFWrapper(df,isData, /*ele_sf_name=*/ "_weight_electron_reco_tightid", /*muo_sf_name=*/ "_weight_muon_tightid_tightiso", /*include_trigger_sf=*/ true);

        // Channel orthogonality selection
        df = df.Filter("(nLep_Sel == 1)", "C2: 1lep_1FJ");

        df = df.Filter("((nMuon_Loose == 1 && nMuon_Tight == 1 && nElectron_Loose == 0 && nElectron_Tight == 0) || "
                       "(nMuon_Loose == 0 && nMuon_Tight == 0 && nElectron_Loose == 1 && nElectron_Tight == 1)) && "
                       "(lepton_pt[0] > 40)");
        Cutflow::Add(df, "C2: 1-lepton selection");

        df = definePerVariationPassFlags(df, "1lep_1FJ", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            const std::string jCountCol  = sfx.empty() ? "njet"     : "njet_"     + sfx;
            return "(" + fjCountCol + " == 1) && (" + jCountCol + " >= 4)";
        });
        df = df.Filter(orPassExpr(df, "1lep_1FJ"), "C3: jet selection (any variation)");
    }

    // 1lep_2FJ - same caveat as 1lep_1FJ (C3 + C4 collapsed).
    else if (channel == "1lep_2FJ"){

        df = VBSTagging(df, "jetNoFJClean");
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_singlelep);
        Cutflow::Add(df, "C1: Trigger selection");

        df = lepSFWrapper(df,isData, /*ele_sf_name=*/ "_weight_electron_reco_tightid", /*muo_sf_name=*/ "_weight_muon_tightid_tightiso", /*include_trigger_sf=*/ true);

        // Channel orthogonality selection
        df = df.Filter("(nLep_Sel == 1)", "C2: 1lep_2FJ");

        df = df.Filter("((nMuon_Loose == 1 && nMuon_Tight == 1 && nElectron_Veto == 0 && nElectron_Loose == 0 && nElectron_Tight == 0) || "
                       "(nMuon_Loose == 0 && nMuon_Tight == 0 && nElectron_Veto == 1 && nElectron_Loose == 1 && nElectron_Tight == 1)) && "
                       "(lepton_pt[0] > 40)");

        Cutflow::Add(df, "C2: 1-lepton selection");

        df = definePerVariationPassFlags(df, "1lep_2FJ", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            const std::string jCountCol  = sfx.empty() ? "njet"     : "njet_"     + sfx;
            return "(" + fjCountCol + " >= 2) && (" + jCountCol + " >= 2)";
        });
        df = df.Filter(orPassExpr(df, "1lep_2FJ"), "C3: jet selection (any variation)");
    }

    // 2lepSS
    else if (channel == "2lepSS"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_multilep);
        Cutflow::Add(df, "C1: Trigger selection");

        df = lepSFWrapper(df, isData, /*ele_sf_name=*/ "_weight_electron_reco_looseid", /*muo_sf_name=*/ "_weight_muon_looseid_looseiso");

        // Channel orthogonality selection
        //TODO implement a same sign requirement
        df = df.Filter("(nLep_Sel == 2)", "C2: 2lepSS");

    }

    // 2lep_1FJ (currently shared between OF and SF)
    else if (channel == "2lep_1FJ"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_multilep);
        Cutflow::Add(df, "C1: Trigger selection");

        df = lepSFWrapper(df, isData, /*ele_sf_name=*/ "_weight_electron_reco_looseid", /*muo_sf_name=*/ "_weight_muon_mediumid_tightiso");

        // Channel orthogonality selection
        df = definePerVariationPassFlags(df, "2lep_1FJ", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            return "(nLep_Sel == 2) && (" + fjCountCol + " == 1)";
        });
        df = df.Filter(orPassExpr(df, "2lep_1FJ"), "C2: 2lep_1FJ");
    }

    // 2lep_2FJ
    else if (channel == "2lep_2FJ"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_multilep);
        Cutflow::Add(df, "C1: Trigger selection");

        df = lepSFWrapper(df, isData, /*ele_sf_name=*/ "_weight_electron_reco_looseid", /*muo_sf_name=*/ "_weight_muon_looseid_looseiso");

        // Channel orthogonality selection
        df = definePerVariationPassFlags(df, "2lep_2FJ", [](const std::string& sfx){
            const std::string fjCountCol = sfx.empty() ? "nfatjet" : "nfatjet_" + sfx;
            return "(nLep_Sel == 2) && (" + fjCountCol + " == 2)";
        });
        df = df.Filter(orPassExpr(df, "2lep_2FJ"), "C2: 2lep_2FJ");
    }


    // 3lep
    else if (channel == "3lep"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_multilep);
        Cutflow::Add(df, "C1: Trigger selection");

        df = lepSFWrapper(df, isData, /*ele_sf_name=*/ "_weight_electron_reco_looseid", /*muo_sf_name=*/ "_weight_muon_mediumid_tightiso");

        // Channel orthogonality selection
        df = df.Filter("(nLep_Sel == 3)", "C2: 3lep");

    }

    // 4lep
    else if (channel == "4lep"){

        df = VBSTagging(df);
        Cutflow::Add(df, "VBS pair candidate found");

        df = TriggerSelections(df,trigger_logic_string_multilep);
        Cutflow::Add(df, "C1: Trigger selection");

        df = lepSFWrapper(df, isData, /*ele_sf_name=*/ "_weight_electron_reco_looseid", /*muo_sf_name=*/ "_weight_muon_looseid_looseiso");

        // Channel orthogonality selection
        df = df.Filter("(nLep_Sel == 4)", "C2: 4lep");

    }

    return df;
}
