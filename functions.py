import os, sys
import awkward as ak
import numpy as np
from argparse import ArgumentParser

def chooseOverlaps():
    OPTIONS = {
        'run2': {
            'DY_M-50': {
                'A': [
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v5_NANOAODSIM",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v2_NANOAODSIM",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v2_NANOAODSIM"
                ],
                'B': [
                    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'WToLNu': {
                'A': [
                    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1_ext2-v1_NANOAODSIM",
                    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext2-v1_NANOAODSIM",
                    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext2-v1_NANOAODSIM",
                    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext2-v1_NANOAODSIM",
                    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1_ext1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'WW': {
                'A': [
                    "WW_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WW_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WW_TuneCP5_13TeV-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WW_TuneCP5_13TeV-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WWTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WWTo4Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WWTo4Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WWTo4Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WWTo4Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'WZ': {
                'A': [
                    "WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZTo1L1Nu2Q_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZTo1L3Nu_4f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ZZ': {
                'A': [
                    "ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo2Q2L_mllmin4p0_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "ZZTo4L_M-1toInf_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo4L_M-1toInf_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo4L_M-1toInf_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo4L_M-1toInf_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "ZZTo4Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZZTo4Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZZTo4Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZZTo4Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ttW': {
                'A': [
                    "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM",
                    "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ttZ': {
                'A': [
                    "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ZH_HToWW': {
                'A': [
                    "HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "HZJ_HToWWTo2L2Nu_ZTo2L_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "VHToNonbb_M125_TuneCP5_13TeV-amcatnloFXFX_madspin_pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ZH_HToBB': {
                'A': [
                    "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "ZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v2_NANOAODSIM",
                    "ZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            },
            'ggZH_HToBB': {
                'A': [
                    "ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v2_NANOAODSIM",
                    "ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ggZH_HToBB_ZToQQ_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ],
                'B': [
                    "ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODAPVv15-150X_mcRun2_asymptotic_preVFP_v1-v1_NANOAODSIM",
                    "ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv15-150X_mcRun2_asymptotic_v1-v1_NANOAODSIM",
                    "ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL17NanoAODv15-150X_mc2017_realistic_v1-v1_NANOAODSIM",
                    "ggZH_HToBB_ZToBB_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1_NANOAODSIM"
                ]
            }
        },
        'run3': {
            'DY_M-50': {
                'A': [
                    "DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4_NANOAODSIM",
                    "DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4_NANOAODSIMSummer24for2025",
                    "DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v6_NANOAODSIM",
                    "DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v6_NANOAODSIMSummer24for2025",
                    "DYto2Tau-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v7_NANOAODSIM",
                    "DYto2Tau-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v7_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'QCD': {
                'A': [
                    "QCD_Bin-PT-1000to1500_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-1000to1500_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-120to170_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-120to170_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-1500to2000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-1500to2000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-170to300_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-170to300_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-2000to2500_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-2000to2500_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-2500to3000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-2500to3000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-3000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-3000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-300to470_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-300to470_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-470to600_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-470to600_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-50to80_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-50to80_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-600to800_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-600to800_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-800to1000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-800to1000_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD_Bin-PT-80to120_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD_Bin-PT-80to120_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "QCD-4Jets_Bin-HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-1000to1200_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-1200to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-1500to2000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-2000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-600to800_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "QCD-4Jets_Bin-HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "QCD-4Jets_Bin-HT-800to1000_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'WToLNu': {
                'A': [
                    "WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3_NANOAODSIMSummer24for2025",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "WtoLNu-4Jets_Bin-1J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-4Jets_Bin-1J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-4Jets_Bin-2J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-4Jets_Bin-2J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-4Jets_Bin-3J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-4Jets_Bin-3J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WtoLNu-4Jets_Bin-4J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WtoLNu-4Jets_Bin-4J_TuneCP5_13p6TeV_madgraphMLM-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'WW': {
                'A': [
                    "WW_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WW_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WWto4Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WWtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'WZ': {
                'A': [
                    "WZ_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZ_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WZtoL3Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZtoL3Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'ZZ': {
                'A': [
                    "ZZ_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZ_TuneCP5_13p6TeV_pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "ZZto2Nu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto2Nu2Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "ZZto4Q-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto4Q-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'QCD_ZZTo4L': {
                'A': [
                    "ZZJJto4L-QCD_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZJJto4L-QCD_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            },
            'EWK_SSWW': {
                'A': [
                    "VBS-SSWW-LL_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "VBS-SSWW-LL_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025",
                    "VBS-SSWW-TL_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "VBS-SSWW-TL_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ],
                'B': [
                    "WWJJto2L2Nu-SS-noTop-EWK_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIM",
                    "WWJJto2L2Nu-SS-noTop-EWK_TuneCP5_13p6TeV_madgraph-pythia8_RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2_NANOAODSIMSummer24for2025"
                ]
            }
        }
    }
    CHOICES = {
        'run2': {
            'DY_M-50'   : 'B',  # B is HT binned, more stats (with B, HT<70 not covered, but neglegible)
            'WToLNu'    : 'B',  # B is HT binned, more stats
            'WW'        : 'B',  # B is exclusive in DM, more stats
            'WZ'        : 'B',  # B is exclusive in DM, more stats
            'ZZ'        : 'B',  # B is exclusive in DM, more stats
            'ttW'       : 'B',  # A does not have 2016preVFP
            'ttZ'       : 'A',  # B only covers LLNuNu (misses ZToQQ)
            'ZH_HToWW'  : 'B',  # A is subset of B
            'ZH_HToBB'  : 'A',  # USE A. Option B only considers ZToBB, which is a subset of ZToQQ.
            'ggZH_HToBB': 'A'   # Same issue as in ZH_HToBB.
        },
        'run3': {
            'DY_M-50'   : 'B',
            'QCD'       : 'B',
            'WToLNu'    : 'A',
            'WW'        : 'B',
            'WZ'        : 'B',
            'ZZ'        : 'B',
            'QCD_ZZTo4L': 'B',
            'EWK_SSWW'  : 'B'
        }
    }
    # Not handled in run2 (3 issues):
    # 1) A: EWKZ2Jets_ZToLL   , B: ZZTo2Q2L + WZTo2Q2L
    # 2) A: EWKZ2Jets_ZToNuNu , B: ZZTo2Nu2Q
    # 3) A: EWKW+-2Jets_WToLNu, B: WWTo1L1Nu2Q + WZTo1L1Nu2Q
    # Currently: As and Bs are both present, so double counting happens.
    # Plan: Check after seeing event distribution, if they are big.
    return {sample for run, groups in OPTIONS.items() for group, opts in groups.items() for opt, samples in opts.items() if opt != CHOICES[run][group] for sample in samples}