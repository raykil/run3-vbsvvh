#include "utils.h"

#include <algorithm>
#include <set>
#include "TFile.h"
#include "TTree.h"

/*
############################################
RDF UTILS
############################################
*/

RNode defineMetadata(RNode df, bool isData = false) {
    if (isData) {df = df.Define("genWeight", []() { return 1.f; }, {});}
    return df.DefinePerSample("xsec", [](unsigned int slot, const RSampleInfo &id) { return id.GetD("xsec");})
        .DefinePerSample("lumi", [](unsigned int slot, const RSampleInfo &id) { return id.GetD("lumi");})
        .DefinePerSample("sumw", [](unsigned int slot, const RSampleInfo &id) { return id.GetD("sumw");})
        .DefinePerSample("kind", [](unsigned int slot, const RSampleInfo &id) { return id.GetS("kind");})
        .DefinePerSample("year", [](unsigned int slot, const RSampleInfo &id) { return id.GetS("year");})
        .DefinePerSample("shortname", [](unsigned int slot, const RSampleInfo &id) { return id.GetS("shortname");})
        .DefinePerSample("name", [](unsigned int slot, const RSampleInfo &id) { return id.GetSampleName();})
        .DefinePerSample("do_ewk_corr", [](unsigned int slot, const RSampleInfo &id) { return id.GetI("do_ewk_corr");})
        .Define("isData", "kind == \"data\"")
        .Define("is2016", "year == \"2016preVFP\" || year == \"2016postVFP\"")
        .Define("is2017", "year == \"2017\"")
        .Define("is2018", "year == \"2018\"")
        .Define("is2022", "year == \"2022Re-recoBCD\" || year == \"2022Re-recoE+PromptFG\"")
        .Define("is2023", "year == \"2023PromptC\" || year == \"2023PromptD\"")
        .Define("is2024", "year == \"2024Prompt\"")
        .Define("is2025", "year == \"2025\"")
        .Define("isRun2", "is2016 || is2017 || is2018")
        .Define("isRun3", "is2022 || is2023 || is2024 || is2025")
        .Define("xsecweight", "isData ? 1 : 1000 * xsec * lumi / sumw")
        .Define("baseweight", "xsecweight * genWeight")
        .Define("weight", "baseweight");

}

// Extract sample kind from the JSON config file
// Returns the kind of the first sample found (assumes all samples in a config have the same kind)
std::string getCategoryFromConfig(const std::string& config_path) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(config_path, pt);

    // Navigate to samples and get the first sample's kind
    for (const auto& sample : pt.get_child("samples")) {
        return sample.second.get<std::string>("metadata.kind");
    }
    return "";
}

BTagEfficiencyMetadata getSingleSampleBTagEfficiencyMetadata(const std::string& config_path) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(config_path, pt);
    const auto &samples = pt.get_child("samples");
    if (samples.size() != 1)
        throw std::runtime_error("--btag_eff requires a config containing exactly one MC sample");
    const auto &sample = *samples.begin();
    return {sample.first, sample.second.get<std::string>("metadata.year")};
}


std::vector<BTagEfficiencyMetadata> getBTagEfficiencyMetadata(const std::string& config_path) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(config_path, pt);
    std::vector<BTagEfficiencyMetadata> metadata;
    std::set<std::pair<std::string, std::string>> seen;
    for (const auto &sample : pt.get_child("samples")) {
        if (sample.second.get<std::string>("metadata.kind") == "data") continue;
        const BTagEfficiencyMetadata entry{
            sample.first, sample.second.get<std::string>("metadata.year")};
        if (seen.emplace(entry.year, entry.sample).second) metadata.push_back(entry);
    }
    return metadata;
}

std::vector<std::string> getMCYearsFromConfig(const std::string& config_path) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(config_path, pt);
    std::vector<std::string> years;
    for (const auto &sample : pt.get_child("samples")) {
        if (sample.second.get<std::string>("metadata.kind") != "data")
            years.push_back(sample.second.get<std::string>("metadata.year"));
    }
    return years;
}

RNode removeDuplicates(RNode df){
    return df.Filter(FilterOnePerKind(), {"run", "luminosityBlock", "event"}, "REMOVED DUPLICATES");
}

RNode applyObjectMask(RNode df, const std::string& maskName, const std::string& objectName) {
    auto columnNames = df.GetColumnNames();
    for (const auto& colName : columnNames) {
        if (colName.starts_with(objectName + "_")) {
            df = df.Redefine(colName, colName + "[" + maskName + "]");
        }
    }
    df = df.Redefine("n" + objectName, "Sum(" + maskName + ")");
    return df;
}

RNode applyObjectMaskNewAffix(RNode df, const std::string &maskName, const std::string &objectName, const std::string &newAffix)
{
    auto columnNames = df.GetColumnNames();
    for (const auto &colName : columnNames)
    {
        if (colName.starts_with(objectName + "_"))
        {
            std::string suffix = colName.substr(objectName.size() + 1);
            // Skip per-source systematic-variation columns. Those carry their own per-
            // variation good-jet masks (Jet_isGood_jes*) and downstream coffea reads them
            // at the all-jet level, so they must NOT be aliased through the nominal mask.
            if (suffix.find("_jes") != std::string::npos
                || suffix.find("_jer") != std::string::npos
                || suffix.find("_jms") != std::string::npos
                || suffix.find("_jmr") != std::string::npos
                || suffix.find("_unclust") != std::string::npos) {
                continue;
            }
            std::string newCol = newAffix + "_" + suffix;
            df = df.Define(newCol, colName + "[" + maskName + "]");
        }
    }
    df = df.Define("n" + newAffix, "Sum(" + maskName + ")");
    return df;
}

/*
############################################
LUMIMASK - GOLDEN JSON
############################################
*/

bool operator< ( const lumiMask::LumiBlockRange& lh, const lumiMask::LumiBlockRange& rh )
{
    return ( lh.run() == rh.run() ) ? ( lh.lastLumi() < rh.firstLumi() ) : lh.run() < rh.run();
}

lumiMask lumiMask::fromJSON(const std::vector<std::string>& files, lumiMask::Run firstRun, lumiMask::Run lastRun)
{
  const bool noRunFilter = ( firstRun == 0 ) && ( lastRun == 0 );

  std::vector<lumiMask::LumiBlockRange> accept;

  for ( const auto& file : files ) {
    boost::property_tree::ptree ptree;
    boost::property_tree::read_json(file, ptree);
    for ( const auto& runEntry : ptree ) {
        const lumiMask::Run run = std::stoul(runEntry.first);
        if ( noRunFilter || ( ( firstRun <= run ) && ( run <= lastRun ) ) ) {
        for ( const auto& lrEntry : runEntry.second ) {
            const auto lrNd = lrEntry.second;
            const lumiMask::LumiBlock firstLumi = std::stoul(lrNd.begin()->second.data());
            const lumiMask::LumiBlock lastLumi  = std::stoul((++lrNd.begin())->second.data());
            accept.emplace_back(run, firstLumi, lastLumi);
        }
        }
    }
  }
  
  return lumiMask(accept);
}

/*
############################################
SELECTION UTILS
############################################
*/

float fdR(float eta1, float phi1, float eta2, float phi2) {
    return ROOT::VecOps::DeltaR(eta1, eta2, phi1, phi2);
}

RVec<float> VdR(const RVec<float>& vec_eta, const RVec<float>& vec_phi, float obj_eta, float obj_phi) {
    RVec<float> out(vec_eta.size());
    if (obj_eta == -999 || obj_phi == -999) {
        std::fill(out.begin(), out.end(), 1.0f);
        return out;
    }
    for (size_t i = 0; i < vec_eta.size(); i++) {
        out[i] = ROOT::VecOps::DeltaR(vec_eta[i], obj_eta, vec_phi[i], obj_phi);
    }
    return out;
}

RVec<float> VVdR(const RVec<float>& vec_eta1, const RVec<float>& vec_phi1, const RVec<float>& vec_eta2, const RVec<float>& vec_phi2) {
    if (vec_eta1.empty()) {
        return RVec<float>();
    }
    if (vec_eta2.empty()) {
        return RVec<float>(vec_eta1.size(), 999.0f);
    }
    RVec<float> out(vec_eta1.size());
    for (size_t i = 0; i < vec_eta1.size(); i++) {
        float mindR = 999.;
        for (size_t j = 0; j < vec_eta2.size(); j++) {
            float dR = ROOT::VecOps::DeltaR(vec_eta1[i], vec_eta2[j], vec_phi1[i], vec_phi2[j]);
            if (dR < mindR) {
                mindR = dR;
            }
        }
        out[i] = mindR;
    }
    return out;
}

float fInvariantMass(float obj1_pt, float obj1_eta, float obj1_phi, float obj1_mass, 
                    float obj2_pt, float obj2_eta, float obj2_phi, float obj2_mass) {
    TLorentzVector obj1, obj2;
    obj1.SetPtEtaPhiM(obj1_pt, obj1_eta, obj1_phi, obj1_mass);
    obj2.SetPtEtaPhiM(obj2_pt, obj2_eta, obj2_phi, obj2_mass);
    return (obj1 + obj2).M();
}

RVec<float> VInvariantMass(const RVec<float>& vec_pt, const RVec<float>& vec_eta, const RVec<float>& vec_phi, 
                          const RVec<float>& vec_mass, float obj_pt, float obj_eta, float obj_phi, float obj_mass) {
    RVec<float> invMass(vec_pt.size());
    TLorentzVector obj1;
    obj1.SetPtEtaPhiM(obj_pt, obj_eta, obj_phi, obj_mass);

    for (size_t i = 0; i < vec_pt.size(); i++) {
        TLorentzVector obj2;
        obj2.SetPtEtaPhiM(vec_pt[i], vec_eta[i], vec_phi[i], vec_mass[i]);
        invMass[i] = (obj1 + obj2).M();
    }
    return invMass;
}

RVec<float> VInvariantPt(const RVec<float>& vec_pt, const RVec<float>& vec_eta, const RVec<float>& vec_phi, 
                        const RVec<float>& vec_mass, float obj_pt, float obj_eta, float obj_phi, float obj_mass) {
    RVec<float> invPt(vec_pt.size());
    TLorentzVector obj1;
    obj1.SetPtEtaPhiM(obj_pt, obj_eta, obj_phi, obj_mass);

    for (size_t i = 0; i < vec_pt.size(); i++) {
        TLorentzVector obj2;
        obj2.SetPtEtaPhiM(vec_pt[i], vec_eta[i], vec_phi[i], vec_mass[i]);
        invPt[i] = (obj1 + obj2).Pt();
    }
    return invPt;
}

RVec<float> VInvariantPhi(const RVec<float>& vec_pt, const RVec<float>& vec_eta, const RVec<float>& vec_phi, 
                         const RVec<float>& vec_mass, float obj_pt, float obj_eta, float obj_phi, float obj_mass) {
    RVec<float> invPhi(vec_pt.size());
    TLorentzVector obj1;
    obj1.SetPtEtaPhiM(obj_pt, obj_eta, obj_phi, obj_mass);

    for (size_t i = 0; i < vec_pt.size(); i++) {
        TLorentzVector obj2;
        obj2.SetPtEtaPhiM(vec_pt[i], vec_eta[i], vec_phi[i], vec_mass[i]);
        invPhi[i] = (obj1 + obj2).Phi();
    }
    return invPhi;
}

RVec<float> VTransverseMass(const RVec<float>& vec_pt, const RVec<float>& vec_phi, float obj_pt, float obj_phi) {
    RVec<float> mt(vec_pt.size());
    for (size_t i = 0; i < vec_pt.size(); i++) {
        mt[i] = std::sqrt(2 * vec_pt[i] * obj_pt * (1 - std::cos(ROOT::VecOps::DeltaPhi(vec_phi[i], obj_phi))));
    }
    return mt;
}

// Return for each ak4 jet, the dR from the closest ak8 jet
RVec<float> dRfromClosestJet(const RVec<float>& ak4_eta, const RVec<float>& ak4_phi, const RVec<float>& ak8_eta, const RVec<float>& ak8_phi) {
    RVec<float> vec_minDR = {};
    for (size_t i = 0; i < ak4_eta.size(); i++)
    {
        float mindR = 999.;
        for (size_t j = 0; j < ak8_eta.size(); j++)
        {
            float dR = ROOT::VecOps::DeltaR(ak4_eta.at(i), ak8_eta.at(j), ak4_phi.at(i), ak8_phi.at(j));
            if (dR < mindR) {
                mindR = dR;
            }
        }
        vec_minDR.push_back(mindR);
    }
    return vec_minDR;
}

RVec<RVec<int>> getVBSPairs(const RVec<int>& goodJets, const RVec<float>& jet_var) {
    if (Sum(goodJets) >= 2) {
        return ROOT::VecOps::Combinations(jet_var, 2);
    } else {
    // Create properly matched return type: vector of vector
        RVec<RVec<int>> result;
        // Add two empty vectors
        result.emplace_back(RVec<int>{-999});
        result.emplace_back(RVec<int>{-999});
        return result;
    }
}

RVec<int> VBS_MaxEtaJJ(RVec<float> Jet_pt, RVec<float> Jet_eta, RVec<float> Jet_phi, RVec<float> Jet_mass) {
    // find pair of jets with max delta eta
    RVec<int> good_jet_idx = {};
    RVec<float> Jet_Pt = {};
    for (size_t i = 0; i < Jet_pt.size(); i++) {
        TLorentzVector jet;
        jet.SetPtEtaPhiM(Jet_pt[i], Jet_eta[i], Jet_phi[i], Jet_mass[i]);
        Jet_Pt.push_back(jet.Pt());
    }
    int Nvbfjet1 = -1;
    int Nvbfjet2 = -1;
    float maxvbfjetdeta = 0;
    for (size_t i = 0; i < Jet_eta.size(); i++) {
        for (size_t j = i+1; j < Jet_eta.size(); j++) {
            float deta = std::abs(Jet_eta[i] - Jet_eta[j]);
            if (deta > maxvbfjetdeta) {
                maxvbfjetdeta = deta;
                Nvbfjet1 = i;
                Nvbfjet2 = j;
            }
        }
    }
    if (Jet_Pt[Nvbfjet1] > Jet_Pt[Nvbfjet2]) {
        good_jet_idx.push_back(Nvbfjet1);
        good_jet_idx.push_back(Nvbfjet2);
    }
    else {
        good_jet_idx.push_back(Nvbfjet2);
        good_jet_idx.push_back(Nvbfjet1);
    }
    return good_jet_idx;
}

RVec<float> VBSBDTInfer(RVec<float> Jet_pt, RVec<float> Jet_eta, RVec<float> Jet_phi, RVec<float> Jet_mass, bool isRun2) {
    if (Jet_pt.size() < 2) {
        return RVec<float>{-1, -1, -1};
    }
    auto combination_idxs = ROOT::VecOps::Combinations(Jet_pt, 2);

    auto jet1_pt = ROOT::VecOps::Take(Jet_pt, combination_idxs[0]);
    auto jet1_eta = ROOT::VecOps::Take(Jet_eta, combination_idxs[0]);
    auto jet1_phi = ROOT::VecOps::Take(Jet_phi, combination_idxs[0]);
    auto jet1_mass = ROOT::VecOps::Take(Jet_mass, combination_idxs[0]);
    auto jet2_pt = ROOT::VecOps::Take(Jet_pt, combination_idxs[1]);
    auto jet2_eta = ROOT::VecOps::Take(Jet_eta, combination_idxs[1]);
    auto jet2_phi = ROOT::VecOps::Take(Jet_phi, combination_idxs[1]);
    auto jet2_mass = ROOT::VecOps::Take(Jet_mass, combination_idxs[1]);
    auto detajj = ROOT::VecOps::abs(jet1_eta - jet2_eta);

    auto pt_m_jj = [](const RVec<float>& jet1_pt, const RVec<float>& jet1_eta, const RVec<float>& jet1_phi, const RVec<float>& jet1_mass, 
                        const RVec<float>& jet2_pt, const RVec<float>& jet2_eta, const RVec<float>& jet2_phi, const RVec<float>& jet2_mass) {
        RVec<float> pt_jj;
        RVec<float> m_jj;
        for (size_t i = 0; i < jet1_pt.size(); ++i) {
            auto v_jj = ROOT::Math::PtEtaPhiMVector(jet1_pt[i], jet1_eta[i], jet1_phi[i], jet1_mass[i]) + ROOT::Math::PtEtaPhiMVector(jet2_pt[i], jet2_eta[i], jet2_phi[i], jet2_mass[i]);
            pt_jj.push_back(v_jj.Pt());
            m_jj.push_back(v_jj.M());
        }
        return std::make_pair(pt_jj, m_jj);
    };

    auto [ptjj, mjj] = pt_m_jj(jet1_pt, jet1_eta, jet1_phi, jet1_mass, jet2_pt, jet2_eta, jet2_phi, jet2_mass);
    auto dphijj = ROOT::VecOps::DeltaPhi(jet1_phi, jet2_phi);

    RVec<float> scores;
    float score;
    for (size_t i = 0; i < mjj.size(); i++) {
        score = bdt.Compute({
                    jet1_pt[i], jet2_pt[i],
                    jet1_eta[i], jet2_eta[i],
                    jet1_phi[i], jet2_phi[i],
                    jet1_mass[i], jet2_mass[i],
                    ptjj[i], detajj[i], 
                    dphijj[i], mjj[i]
                })[0];
        scores.push_back(score);
    }
    auto max_score_idx = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
    if (scores.size() > 0) {
        return RVec<float>{static_cast<float>(combination_idxs[0][max_score_idx]),
                         static_cast<float>(combination_idxs[1][max_score_idx]),
                         scores[max_score_idx]};
    }
    return RVec<float>{-1, -1, -1};
}

// VBS tagging on the subset of jets selected by `isGood`, with the returned pair
// indices mapped back into the all-jet (NanoAOD Jet_*) frame, so they dereference
// Jet_pt_<sfx>/Jet_eta/... directly without needing the mask downstream.
// Returns {jet1_JetIdx, jet2_JetIdx, score}; {-1, -1, -999} if fewer than 2 good jets
// (score sentinel deliberately outside the BDT output range).
RVec<float> VBSBDTInferMasked(const RVec<int>& isGood, const RVec<float>& Jet_pt, const RVec<float>& Jet_eta, const RVec<float>& Jet_phi, const RVec<float>& Jet_mass, bool isRun2) {
    auto idx = ROOT::VecOps::Nonzero(isGood);
    if (idx.size() < 2) {
        return RVec<float>{-1, -1, -999};
    }
    auto pair = VBSBDTInfer(ROOT::VecOps::Take(Jet_pt, idx), ROOT::VecOps::Take(Jet_eta, idx),
                            ROOT::VecOps::Take(Jet_phi, idx), ROOT::VecOps::Take(Jet_mass, idx), isRun2);
    if (pair[0] < 0) {
        return RVec<float>{-1, -1, -999};
    }
    return RVec<float>{static_cast<float>(idx[static_cast<size_t>(pair[0])]),
                       static_cast<float>(idx[static_cast<size_t>(pair[1])]),
                       pair[2]};
}

/*
############################################
SNAPSHOT
############################################
*/

std::string setOutputDirectory(const std::string &outdir, bool spanet_training) {
    std::string output_dir = "";
    if (spanet_training) {
        output_dir = outdir + "/spanet_training/";
    }
    else {
        output_dir = outdir;
    }

    std::filesystem::path directory_path(output_dir);
    // Check if the directory exists
    if (std::filesystem::exists(directory_path)) {
        std::cerr << "Output directory: " << directory_path << std::endl;
    }
    // Try to create the directory and any missing parent directories
    else if (std::filesystem::create_directories(directory_path)) {
        std::cout << "Created output directory : " << directory_path << std::endl;
    }
    else {
        std::cerr << "Failed to create output directory: " << directory_path << std::endl;
        std::exit(EXIT_FAILURE); 
    }

    return directory_path;
}

void saveSnapshot(RNode df, const std::string &outputDir, const std::string &outputFileName, bool isSig, bool dumpInput, bool storeHLT)
{
    auto ColNames = df.GetDefinedColumnNames();
    std::vector<std::string> final_variables;
    final_variables.push_back("run");
    final_variables.push_back("luminosityBlock");
    final_variables.push_back("event");

    // Drop internal RDF helpers ("_*") and the raw lepton collections (Electron_, Muon_)
    // — leptons live under the lowercase aliases (electron_, muon_) defined by the
    // selection step. Capital Jet_* / FatJet_* are kept in full so downstream coffea can
    // construct per-variation jet collections from the all-jet-level attributes plus the
    // per-variation good-jet booleans (Jet_isGood_jes*, FatJet_isGood_jes*) defined in
    // selections.cpp. The lowercase jet_* / fatjet_* nominal-good-jets aliases are also
    // kept (existing analysis code reads through them).
    for (auto &&ColName : ColNames) {
        if (ColName.starts_with("_")) continue;
        if (ColName.starts_with("Electron_") || ColName.starts_with("Muon_")) continue;
        final_variables.push_back(ColName);
    }

    // Pull in the raw NanoAOD capital Jet_* / FatJet_* attributes (which are not in the
    // Defined-columns list above). Keep all of them — option-A storage layout.
    auto allColNames = df.GetColumnNames();
    for (auto &&col : allColNames) {
        if ((col.starts_with("Jet_") || col.starts_with("FatJet_")) &&
            std::find(final_variables.begin(), final_variables.end(), col) == final_variables.end()) {
            final_variables.push_back(col);
        }
    }

    // Optionally store HLT branches from input NanoAOD, providing default values
    // for branches that may not exist in all files of a multi-file chain
    if (storeHLT) {
        auto allColNames = df.GetColumnNames();
        for (auto &&colName : allColNames) {
            if (colName.starts_with("HLT_") &&
                std::find(final_variables.begin(), final_variables.end(), colName) == final_variables.end()) {
                df = df.DefaultValueFor(colName, (bool)false);
                final_variables.push_back(colName);
            }
        }
    }

    if (isSig) {
        final_variables.push_back("LHEReweightingWeight");
        // ROOT creates the count branch automatically for vector branches.
        final_variables.push_back("LHEPdfWeight");
    }

    // store all columns from input nanoAOD tree
    if (dumpInput) {
        auto nanoColNames = df.GetColumnNames();
        for (auto &&colName : nanoColNames) {
            // Skip internal "_"-prefixed helper columns: input NanoAOD branches never start with
            // "_", and some helpers (e.g. the Type-1 MET blocks/pairs) have no ROOT dictionary.
            if (colName.starts_with("_")) continue;
            if ((std::find(final_variables.begin(), final_variables.end(), colName) == final_variables.end()) &&
                (colName.find("HLT") == std::string::npos) && (colName.find("L1") == std::string::npos)) {
                final_variables.push_back(colName);
            }
        }
    }

    std::string outputFile = outputDir + "/" + outputFileName + ".root";
    df.Snapshot("Events", outputFile, final_variables);

    // RDataFrame::Snapshot() in multi-threaded mode does not write a TTree when
    // 0 events pass the filters, producing a ROOT file with no keys.  Ensure the
    // output always contains an "Events" TTree so downstream code can open it.
    {
        TFile f(outputFile.c_str(), "UPDATE");
        if (!f.Get("Events")) {
            TTree t("Events", "Events");
            t.Write();
        }
        f.Close();
    }

    std::cout << " -> Stored output file: " << outputFile << std::endl;
}


void saveSpanetSnapshot(RNode df, const std::string &outputDir, const std::string &outputFileName)
{
    auto ColNames = df.GetColumnNames();
    std::vector<std::string> final_variables;
    final_variables.push_back("event");

    for (auto &&ColName : ColNames) {
        if (ColName.starts_with("jet_") ||
            ColName.starts_with("fatjet_") ||
            ColName.starts_with("PuppiMET_") ||
            ColName.starts_with("GenPart_") ||
            ColName.starts_with("lepton_") ||
            ColName.starts_with("gen_") ||
            ColName.starts_with("truth_")) {
                final_variables.push_back(ColName);
            }
    }

    std::string outputFile = outputDir + "/" + outputFileName + "_spanet_training_data.root";
    df.Snapshot("Events", outputFile, final_variables);

    {
        TFile f(outputFile.c_str(), "UPDATE");
        if (!f.Get("Events")) {
            TTree t("Events", "Events");
            t.Write();
        }
        f.Close();
    }

    std::cout << " -> Stored output file: " << outputFile << std::endl;
}
