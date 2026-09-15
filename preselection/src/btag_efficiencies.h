#ifndef BTAG_EFFICIENCIES_H
#define BTAG_EFFICIENCIES_H

#pragma once

#include <string>

#include "ROOT/RDataFrame.hxx"

using RNode = ROOT::RDF::RNode;

// Write schema-v2 signed nominal-MC-weighted selected-AK4 jet yields for the
// ordered L/M/T/XT/XXT working points and their exclusive categories. The
// converter combines job outputs before calculating efficiencies.
void saveBTagEfficiencyHistograms(RNode df, const std::string &output_dir,
                                  const std::string &output_name,
                                  const std::string &channel,
                                  const std::string &year,
                                  const std::string &sample,
                                  int nslots);

#endif
