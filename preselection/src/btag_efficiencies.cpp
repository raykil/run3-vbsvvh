#include "btag_efficiencies.h"
#include "btag_settings.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ROOT/RVec.hxx"
#include "TFile.h"
#include "TH2D.h"
#include "TNamed.h"

namespace {

constexpr std::array<const char *, 3> kFlavors = {"b", "c", "light"};
constexpr std::array<std::string_view, 11> kKinds = {
    "den", kBTagInclusiveWorkingPoints[0], kBTagInclusiveWorkingPoints[1],
    kBTagInclusiveWorkingPoints[2], kBTagInclusiveWorkingPoints[3], kBTagInclusiveWorkingPoints[4],
    kBTagExclusiveCategories[1], kBTagExclusiveCategories[2], kBTagExclusiveCategories[3],
    kBTagExclusiveCategories[4], kBTagExclusiveCategories[0]
};
constexpr std::array<double, 11> kPtBins = {15., 20., 30., 50., 70., 100.,
                                             140., 200., 300., 600., 1000.};
constexpr std::array<double, 5> kEtaBins = {-2.5, -0.9, 0., 0.9, 2.5};

int flavorIndex(int hadron_flavor) {
    const int flavor = std::abs(hadron_flavor);
    if (flavor == 5) return 0;
    if (flavor == 4) return 1;
    return 2;
}

struct HistogramSet {
    std::array<std::array<std::unique_ptr<TH2D>, kKinds.size()>, kFlavors.size()> histograms;

    HistogramSet(const std::string &prefix) {
        for (std::size_t flavor = 0; flavor < kFlavors.size(); ++flavor) {
            for (std::size_t kind = 0; kind < kKinds.size(); ++kind) {
                const std::string name = prefix + "btag_" + kFlavors[flavor] + "_" + std::string(kKinds[kind]);
                auto hist = std::make_unique<TH2D>(name.c_str(), name.c_str(),
                                                   kPtBins.size() - 1, kPtBins.data(),
                                                   kEtaBins.size() - 1, kEtaBins.data());
                hist->SetDirectory(nullptr);
                hist->Sumw2();
                histograms[flavor][kind] = std::move(hist);
            }
        }
    }

    void fill(float pt, float eta, int hadron_flavor, const std::array<bool, 5> &passed,
              double weight,
              double max_abs_eta) {
        // BTV AK4 calibrations are defined in the central-jet region.  The
        // selected collection can include forward jets, which must not enter
        // these efficiencies or their SF application.
        if (std::abs(eta) >= max_abs_eta) return;
        for (std::size_t wp = 1; wp < passed.size(); ++wp) {
            if (passed[wp] && !passed[wp - 1])
                throw std::runtime_error("Non-nested UParTAK4 working-point flags: XXT => XT => T => M => L was violated");
        }
        const int flavor = flavorIndex(hadron_flavor);
        histograms[flavor][0]->Fill(pt, eta, weight); // denominator
        for (std::size_t wp = 0; wp < passed.size(); ++wp)
            if (passed[wp]) histograms[flavor][wp + 1]->Fill(pt, eta, weight);

        // The exclusive categories are filled exactly once.  XXT is also the
        // tightest inclusive state, so its existing histogram is reused.
        if (passed[4]) { /* XXT was filled once above as the inclusive state. */ }
        else if (passed[3]) histograms[flavor][9]->Fill(pt, eta, weight);  // XTnotXXT
        else if (passed[2]) histograms[flavor][8]->Fill(pt, eta, weight);  // TnotXT
        else if (passed[1]) histograms[flavor][7]->Fill(pt, eta, weight);  // MnotT
        else if (passed[0]) histograms[flavor][6]->Fill(pt, eta, weight);  // LnotM
        else histograms[flavor][10]->Fill(pt, eta, weight);                // N
    }

    void add(const HistogramSet &other) {
        for (std::size_t flavor = 0; flavor < kFlavors.size(); ++flavor)
            for (std::size_t kind = 0; kind < kKinds.size(); ++kind)
                histograms[flavor][kind]->Add(other.histograms[flavor][kind].get());
    }

    void write() const {
        for (const auto &flavor : histograms)
            for (const auto &hist : flavor)
                hist->Write();
    }
};

} // namespace

void saveBTagEfficiencyHistograms(RNode df, const std::string &output_dir,
                                  const std::string &output_name,
                                  const std::string &channel,
                                  const std::string &year,
                                  const std::string &sample,
                                  int nslots) {
    if (nslots < 1) nslots = 1;

    std::vector<std::unique_ptr<HistogramSet>> slot_histograms;
    slot_histograms.reserve(nslots);
    for (int slot = 0; slot < nslots; ++slot)
        slot_histograms.emplace_back(std::make_unique<HistogramSet>("slot" + std::to_string(slot) + "_"));

    df.ForeachSlot(
        [&slot_histograms, &year](unsigned int slot, const ROOT::VecOps::RVec<float> &pt,
                           const ROOT::VecOps::RVec<float> &eta,
                           const ROOT::VecOps::RVec<unsigned char> &hadron_flavor,
                           const ROOT::VecOps::RVec<bool> &loose,
                           const ROOT::VecOps::RVec<bool> &medium,
                           const ROOT::VecOps::RVec<bool> &tight,
                           const ROOT::VecOps::RVec<bool> &extra_tight,
                           const ROOT::VecOps::RVec<bool> &extra_extra_tight,
                           double baseweight) {
            if (slot >= slot_histograms.size())
                throw std::runtime_error("RDataFrame used more b-tag histogram slots than allocated");
            if (pt.size() != eta.size() || pt.size() != hadron_flavor.size() ||
                pt.size() != loose.size() || pt.size() != medium.size() ||
                pt.size() != tight.size() || pt.size() != extra_tight.size() ||
                pt.size() != extra_extra_tight.size())
                throw std::runtime_error("Selected AK4 jet branches have inconsistent sizes");
            for (std::size_t jet = 0; jet < pt.size(); ++jet) {
                const std::array<bool, 5> passed = {
                    loose[jet], medium[jet], tight[jet], extra_tight[jet], extra_extra_tight[jet]
                };
                slot_histograms[slot]->fill(pt[jet], eta[jet], hadron_flavor[jet], passed,
                                            baseweight, bTagMaxAbsEta(year));
            }
        },
        {"jet_pt", "jet_eta", "jet_hadronFlavour", "jet_isLooseBTag", "jet_isMediumBTag",
         "jet_isTightBTag", "jet_isExtraTightBTag", "jet_isExtraExtraTightBTag", "baseweight"});

    HistogramSet merged("");
    for (const auto &histograms : slot_histograms)
        merged.add(*histograms);

    std::filesystem::create_directories(output_dir);
    const std::string path = output_dir + "/" + output_name + "_btag_eff.root";
    TFile output(path.c_str(), "RECREATE");
    if (output.IsZombie())
        throw std::runtime_error("Could not create b-tag efficiency output: " + path);

    TNamed("btag_eff_channel", channel.c_str()).Write();
    TNamed("btag_eff_year", year.c_str()).Write();
    TNamed("btag_eff_sample", sample.c_str()).Write();
    TNamed("btag_eff_working_points", "L,M,T,XT,XXT").Write();
    TNamed("btag_eff_schema_version", "2").Write();
    TNamed("btag_eff_format", "signed baseweight selected-jet yields; schema v2; efficiencies must be calculated after merging").Write();
    merged.write();
    output.Close();
}
