# corrections

## The `from_jsonpog-integration` directory

The corrections in this sub directory come from `https://gitlab.cern.ch/cms-crossPOG/jsonpog-integration.git`.

### Muons

We get the Muon SFs from this repo, as outlined in the Muon documentation: `https://muon-wiki.docs.cern.ch/guidelines/corrections/`.

The files were obtained via cloning the `jsonpog-integration` repo and then copying the relevant json files, with the naming as follows:
```
cp jsonpog-integration/POG/MUO/2016preVFP_UL/muon_Z.json.gz     from_jsonpog-integration/MUO__2016preVFP_UL__muon_Z.json.gz
cp jsonpog-integration/POG/MUO/2016postVFP_UL/muon_Z.json.gz    from_jsonpog-integration/MUO__2016postVFP_UL__muon_Z.json.gz
cp jsonpog-integration/POG/MUO/2017_UL/muon_Z.json.gz           from_jsonpog-integration/MUO__2017_UL__muon_Z.json.gz
cp jsonpog-integration/POG/MUO/2018_UL/muon_Z.json.gz           from_jsonpog-integration/MUO__2018_UL__muon_Z.json.gz

cp jsonpog-integration/POG/MUO/2022_Summer22/muon_Z.json.gz     from_jsonpog-integration/MUO__2022_Summer22__muon_Z.json.gz
cp jsonpog-integration/POG/MUO/2022_Summer22EE/muon_Z.json.gz   from_jsonpog-integration/MUO__2022_Summer22EE__muon_Z.json.gz
cp jsonpog-integration/POG/MUO/2023_Summer23/muon_Z.json.gz     from_jsonpog-integration/MUO__2023_Summer23__muon_Z.json.gz
cp jsonpog-integration/POG/MUO/2023_Summer23BPix/muon_Z.json.gz from_jsonpog-integration/MUO__2023_Summer23BPix__muon_Z.json.gz
cp jsonpog-integration/POG/MUO/2024_Summer24/muon_Z.json.gz     from_jsonpog-integration/MUO__2024_Summer24__muon_Z.json.gz
```
