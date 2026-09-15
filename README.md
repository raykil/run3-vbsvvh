# Vector Boson Scattering Analysis Framework

Repository based on RDF for VBS analysis. 
See [preselection](https://github.com/cmstas/run3-vbsvvh/tree/main/preselection#preselection-framework) for preprocessing, and [plotting](https://github.com/cmstas/run3-vbsvvh/tree/main/plotter#plotting-script-documentation) (or [ewkcoffea](https://github.com/cmstas/ewkcoffea)) for histogramming.

##
- RDF are produced by running `preselection/run_rdf.py`.
    - Input files are found in `preselection/etc/input_sample_jsons`. 
    - Selections are found in `preselection/src/selections.cpp`.
- Currently, RDFs are saved in `/groups/cjessop/users/jkil/HVV_2L_RDF` (copied from `/eos/user/r/rband/HVV2LRDF/` in lxplus.)
