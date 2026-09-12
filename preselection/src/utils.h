#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <limits>
#include <filesystem>

#include "tabulate.hpp"

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

#include "TLorentzVector.h"

#include "TMVA/RInferenceUtils.hxx"
#include "TMVA/RReader.hxx"
#include "TMVA/RBDT.hxx"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "TString.h"

using RNode = ROOT::RDF::RNode;
using ROOT::VecOps::RVec;
using ROOT::RDF::RSampleInfo;

/*
############################################
DEFINE METADATA
############################################
*/

RNode defineMetadata(RNode df, bool isData);
std::string getCategoryFromConfig(const std::string& config_path);

struct BTagEfficiencyMetadata {
    std::string sample;
    std::string year;
};
BTagEfficiencyMetadata getSingleSampleBTagEfficiencyMetadata(const std::string& config_path);
std::vector<BTagEfficiencyMetadata> getBTagEfficiencyMetadata(const std::string& config_path);
std::vector<std::string> getMCYearsFromConfig(const std::string& config_path);

class FilterOnePerKind {
    std::unordered_set<size_t> _seenCategories;
public:
    bool operator()(unsigned int run, unsigned int luminosityBlock, unsigned long long event) {
        std::hash<std::string> categoryHasher;
        std::string eventStr = std::to_string(run) + "," + std::to_string(luminosityBlock) + "," + std::to_string(event);
        size_t category = categoryHasher(eventStr);
        {
        // build char category from run, luminosityBlock, event
        R__READ_LOCKGUARD(ROOT::gCoreMutex); // many threads can take a read lock concurrently
        if (_seenCategories.count(category) == 1)
            return false;
        }
        // if we are here, `category` was not already in _seenCategories
        R__WRITE_LOCKGUARD(ROOT::gCoreMutex); // only one thread at a time can take the write lock
        _seenCategories.insert(category);
        return true;
    }
};

RNode removeDuplicates(RNode df);
RNode applyObjectMask(RNode df, const std::string& maskName, const std::string& objectName);
RNode applyObjectMaskNewAffix(RNode df, const std::string &maskName, const std::string &objectName, const std::string &newAffix);

// Define or redefine a column depending on whether it already exists in the dataframe.
// Useful when a function may be called multiple times with different arguments (e.g. AK4JetsSelection).
template <typename F>
RNode DefineOrRedefine(RNode df_, const std::string& colName, F func, const std::vector<std::string>& colArgs)
{
    auto colNames = df_.GetColumnNames();
    if (std::find(colNames.begin(), colNames.end(), colName) != colNames.end())
        return df_.Redefine(colName, func, colArgs);
    else
        return df_.Define(colName, func, colArgs);
}
inline RNode DefineOrRedefine(RNode df_, const std::string& colName, const std::string& expression)
{
    auto colNames = df_.GetColumnNames();
    if (std::find(colNames.begin(), colNames.end(), colName) != colNames.end())
        return df_.Redefine(colName, expression);
    else
        return df_.Define(colName, expression);
}

/*
############################################
LUMIMASK
############################################
*/

class lumiMask {
public:
    using Run = unsigned int;
    using LumiBlock = unsigned int;

    class LumiBlockRange {
    public:
        LumiBlockRange(Run run, LumiBlock firstLumi, LumiBlock lastLumi) : m_run(run), m_firstLumi(firstLumi), m_lastLumi(lastLumi ? lastLumi : std::numeric_limits<LumiBlock>::max()) {}
        Run run() const { return m_run; }
        LumiBlock firstLumi() const { return m_firstLumi; }
        LumiBlock lastLumi () const { return m_lastLumi ; }
    private:
        Run m_run;
        LumiBlock m_firstLumi;
        LumiBlock m_lastLumi;
    };

    explicit lumiMask(const std::vector<LumiBlockRange>& accept) : m_accept(accept) {
        std::sort(m_accept.begin(), m_accept.end());
    }

    double accept(Run run, LumiBlock lumi) const { 
        return std::binary_search(m_accept.begin(), m_accept.end(), LumiBlockRange(run, lumi, lumi)); 
    }

    static lumiMask fromJSON(const std::vector<std::string>& fileNames, lumiMask::Run firstRun=0, lumiMask::Run lastRun=0);

private:
    std::vector<LumiBlockRange> m_accept;
};

bool operator< ( const lumiMask::LumiBlockRange& lh, const lumiMask::LumiBlockRange& rh );


/*
############################################
SELECTION UTILS
############################################
*/

float fdR(float eta1, float phi1, float eta2, float phi2);
float fInvariantMass(float pt1, float eta1, float phi1, float mass1, float pt2, float eta2, float phi2, float mass2);
RVec<float> VdR(const RVec<float>& vec_eta, const RVec<float>& vec_phi, float obj_eta, float obj_phi);
RVec<float> VVdR(const RVec<float>& vec_eta1, const RVec<float>& vec_phi1, const RVec<float>& vec_eta2, const RVec<float>& vec_phi2);
RVec<float> VInvariantMass(const RVec<float>& vec_pt, const RVec<float>& vec_eta, const RVec<float>& vec_phi, const RVec<float>& vec_mass, float obj_pt, float obj_eta, float obj_phi, float obj_mass);
RVec<float> VInvariantPt(const RVec<float>& vec_pt, const RVec<float>& vec_eta, const RVec<float>& vec_phi, const RVec<float>& vec_mass, float obj_pt, float obj_eta, float obj_phi, float obj_mass);
RVec<float> VInvariantPhi(const RVec<float>& vec_pt, const RVec<float>& vec_eta, const RVec<float>& vec_phi, const RVec<float>& vec_mass, float obj_pt, float obj_eta, float obj_phi, float obj_mass);
RVec<float> VTransverseMass(const RVec<float>& vec_pt, const RVec<float>& vec_phi, float obj_pt, float obj_phi);
RVec<float> dRfromClosestJet(const RVec<float>& ak4_eta, const RVec<float>& ak4_phi, const RVec<float>& ak8_eta, const RVec<float>& ak8_phi);

RVec<RVec<int>> getVBSPairs(const RVec<int>& goodJets, const RVec<float>& jetPt);
RVec<int> VBS_MaxEtaJJ(RVec<float> Jet_pt, RVec<float> Jet_eta, RVec<float> Jet_phi, RVec<float> Jet_mass);

RVec<float> VBSBDTInfer(RVec<float> Jet_pt, RVec<float> Jet_eta, RVec<float> Jet_phi, RVec<float> Jet_mass, bool isRun2);
RVec<float> VBSBDTInferMasked(const RVec<int>& isGood, const RVec<float>& Jet_pt, const RVec<float>& Jet_eta, const RVec<float>& Jet_phi, const RVec<float>& Jet_mass, bool isRun2);
const static TMVA::Experimental::RBDT bdt("VBS BDT", "bdt/BDT_Weights.root");

/*
############################################
SNAPSHOT
############################################
*/
std::string setOutputDirectory(const std::string &outdir, bool spanet_training);
void saveSnapshot(RNode df, const std::string &outputDir, const std::string &outputFileName, bool isSig, bool dumpInput, bool storeHLT = false);
void saveSpanetSnapshot(RNode df, const std::string &outputDir, const std::string &outputFileName);

#endif
