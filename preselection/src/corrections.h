#ifndef CORRECTIONS_H
#define CORRECTIONS_H

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RVec.hxx"

#include "correction.h"

using correction::CorrectionSet;
using RNode = ROOT::RDF::RNode;
using ROOT::VecOps::RVec;

/*
############################################
CORRECTION DATA REGISTRY
############################################

All correctionlib payloads (CorrectionSet::from_file) and the era-keyed
tag/name tables live in corrections.cpp — see the CORRECTION-SET REGISTRY
section there. Single definition point, loaded lazily on first use.
To add or re-pin an era, edit the JERC era table in corrections.cpp.
*/

/*
############################################
B-TAGGING WORKING POINTS
############################################
*/
RVec<bool> isbTagLoose(std::string year, RVec<float> btag_score);
RVec<bool> isbTagMedium(std::string year, RVec<float> btag_score);
RVec<bool> isbTagTight(std::string year, RVec<float> btag_score);
RVec<bool> isbTagExtraTight(std::string year, RVec<float> btag_score);
RVec<bool> isbTagExtraExtraTight(std::string year, RVec<float> btag_score);

/*
############################################
MET CORRECTIONS
############################################
*/

RNode applyMETPhiCorrections(RNode df, bool isData);
RNode applyMETUnclusteredVariations(RNode df, bool isData);

/*
############################################
JET MASS SCALE (JMS) AND RESOLUTION (JMR) — generic, per-mass-column
############################################

Convention:
  - JMS  : *additive* shift on the target mass column, in GeV. Central = 0.0 GeV.
  - JMR  : *multiplicative* width factor wrt the gen reference mass. Central = 1.0.

applyJetMassScale / applyJetMassResolution (corrections.cpp) are column-agnostic — the
mass branch and (for JMR) the gen mass + gen-index branches are passed in. This
analysis intends to calibrate on the GloParT regressed mass, not FatJet_msoftdrop,
and the calibration is not yet derived — the helpers
are currently unwired in applyMCCorrections. Identity-valued placeholder maps
(a template for the future calibration) live in corrections.cpp.
*/
RNode applyJetMassScale(const std::unordered_map<std::string, float>& shift_map,
                     const std::string& mass_col, RNode df);
RNode applyJetMassResolution(const std::unordered_map<std::string, float>& factor_map,
                          const std::unordered_map<std::string, float>& sigma_rel_map,
                          const std::string& mass_col,
                          const std::string& gen_mass_col,
                          const std::string& gen_idx_col,
                          RNode df);

/*
############################################
JET ENERGY CORRECTIONS
############################################

The pinned correction files and per-era JEC/JER tag strings are defined by the
JERC era table in corrections.cpp (see the pinning rationale documented there).
*/

// Full Type-1 PuppiMET rebuild from RawPuppiMET over the merged AK4 list (Jet + CorrT1METJet)
// with muon subtraction, per the JERC recipe. Writes met_pt / met_phi (nominal) and, on MC with
// systematics enabled, met_pt_<sfx> / met_phi_<sfx> for each JES and JER variation. Must run AFTER
// applyJetEnergyCorrections / applyJetEnergyResolution / applyJESVariations (it consumes
// Jet_jerFactor and the _Jet_var_<sfx> columns and re-evaluates the JEC compound) and after
// _Jet_rawpt = (1-Jet_rawFactor)*Jet_pt has been defined on the pristine NanoAOD jets.
RNode applyType1MET(RNode df, bool isData);

// Nominal JEC: removes the JEC stored in NanoAOD (via Jet_rawFactor) and re-applies the
// latest L1FastJet * L2Relative * L3Absolute (* L2L3Residual for data) compound from
// jet_jerc.json.gz. MET is NOT propagated here — it is rebuilt in applyType1MET.
RNode applyJetEnergyCorrections(const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
                                const std::unordered_map<std::string, std::string>& jec_prefix_map,
                                const std::unordered_map<std::string, std::string>& jec_suffix_map,
                                RNode df, bool isData);

// JES uncertainty driver — per-source, suffixed-branch output, official JERC ordering:
// the shift (1 ± u) is applied to the JEC (PRE-smearing) pt, and JER is re-evaluated on
// the shifted pt with the same event seed as the nominal smearing. Must run after the
// nominal applyJet/FatJetEnergyCorrections + apply*EnergyResolution (it consumes their
// pre-smear snapshot columns). Emits all 11 Regrouped V2 sources × {Up, Dn} = 22
// variations on AK4 + AK8, correlated between the two algorithms (same sources/tags).
//
// Writes Jet_pt_<suffix>, Jet_mass_<suffix> and the per-jet full variation factor
// _Jet_var_<suffix> (consumed by applyType1MET for the per-variation MET), plus
// FatJet_pt_<suffix>, FatJet_mass_<suffix> (NOT FatJet_msoftdrop — that has its own JEC
// recipe; no MET propagation from AK8 because Type-I MET is built from AK4 only).
//
// <suffix> = "jes" + column_label + direction, e.g. "jesAbsoluteUp", "jesAbsoluteYearDn".
RNode applyJESVariations(RNode df);

// Enables the JES/JER variation branches; called with false (the default) the suffix
// accessors below return empty and applyJESVariations is a no-op. Nominal-only is the
// default because most channels estimate their background from data, so background MC
// does not need the variations. Pass --systs to store them.
void setStoreSysts(bool v);

// ---------------------------------------------------------------------------------------
// DEBUGGING ONLY — jet veto map "compute but do not apply" mode.
//
// Called with false (via --no_jetveto).
// *** NEVER use this for an analysis production. *** Output produced with --no_jetveto is
// NOT veto-map corrected: it retains jets in known-bad (eta, phi) regions and, in Run 3,
// events carrying the spurious MET the veto exists to remove. It is for the jet veto map
// validation study only.
void setApplyJetVetoMaps(bool v);
bool applyJetVetoMapsEnabled();
// ---------------------------------------------------------------------------------------

// Public accessors for the variation suffixes. All return empty when
// setStoreSysts(false) has been called.
//   jesVariationSuffixes:       the 22 JES suffixes ("jesAbsoluteUp", ..., "jesRelativeSampleYearDn")
//   jerVariationSuffixes:       {"jerUp", "jerDn"}
//   kinematicVariationSuffixes: JES + JER combined — the full list of suffixes for which
//                               <collection>_pt_<sfx>/_mass_<sfx> (+ met_pt_<sfx>) may exist.
//                               Consumers building per-variation selections should loop over this.
//   unclusteredVariationSuffixes: {"metunclUp", "metunclDn"} — the MET-only UES suffixes.
std::vector<std::string> jesVariationSuffixes();
std::vector<std::string> jerVariationSuffixes();
std::vector<std::string> kinematicVariationSuffixes();
std::vector<std::string> unclusteredVariationSuffixes();

// JER hybrid smearing (MC only). Defines Jet_jerFactor and redefines Jet_pt/Jet_mass to the
// nominal-smeared values. With storeVariations, also writes the ±1σ SF variation branches
// Jet_jerFactor_jerUp/Dn and Jet_pt/mass_jerUp/Dn.
RNode applyJetEnergyResolution(const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
                               const std::unordered_map<std::string, correction::CorrectionSet>& cset_jer_smear,
                               const std::unordered_map<std::string, std::string>& jer_res_map,
                               const std::unordered_map<std::string, std::string>& jer_sf_map,
                               const std::unordered_map<std::string, std::string>& jer_unc_map,
                               RNode df, bool storeVariations);

// AK8 (FatJet_*) variants — same recipe as AK4 but reading FatJet_* / GenJetAK8_* branches
// and the AK8PFPuppi compound from fatJet_jerc.json.gz.
RNode applyFatJetEnergyCorrections(const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
                                   const std::unordered_map<std::string, std::string>& jec_prefix_map,
                                   const std::unordered_map<std::string, std::string>& jec_suffix_map,
                                   RNode df, bool isData);

RNode applyFatJetEnergyResolution(const std::unordered_map<std::string, correction::CorrectionSet>& cset_jerc,
                                  const std::unordered_map<std::string, correction::CorrectionSet>& cset_jer_smear,
                                  const std::unordered_map<std::string, std::string>& jer_res_map,
                                  const std::unordered_map<std::string, std::string>& jer_sf_map,
                                  const std::unordered_map<std::string, std::string>& jer_unc_map,
                                  RNode df, bool storeVariations);

/*
############################################
JET VETO MAPS
############################################
*/
RNode applyJetVetoMaps(RNode df);

/*
############################################
ELECTRON SCALE AND SMEARING CORRECTIONS
############################################
*/
RNode applyElectronScaleAndSmearing(RNode df, bool isData);

/*
############################################
Others
############################################
*/

RNode HEMCorrection(RNode df, bool isData);

RNode applyDataCorrections(RNode df_);
RNode applyMCCorrections(RNode df_);

#endif // CORRECTIONS_H
