# Third-party licenses and attribution

| Component | Included scope | License | Notice |
|---|---|---|---|
| NeuraChip/NeuraSim artifact | Marked production driver/runtime files | CC BY 4.0 | Kaustubh Shivdikar and Nicolas Bohm Agostini; DOI `10.5281/zenodo.10896280`; modifications described in `NOTICE` |
| DRAMsim3 | Pruned source snapshot and one HBM2 configuration | MIT | Full text: `third_party/DRAMsim3-1.0.0/LICENSE` |
| fmt | Headers vendored with DRAMsim3 | BSD 2-Clause | Full text: `third_party/DRAMsim3-1.0.0/ext/fmt/LICENSE.rst` |
| INIReader/inih | Headers vendored with DRAMsim3 | BSD 3-Clause | Full text: `third_party/DRAMsim3-1.0.0/ext/headers/INIHLICENSE.txt` |
| JSON for Modern C++ | `json.hpp` vendored with DRAMsim3 | MIT | Copyright and full MIT notice embedded in `json.hpp` |

The DRAMsim3 snapshot is pruned to the files needed by the simulator integration;
upstream tests, traces, thermal solver, command-line program, and unrelated
memory configurations are intentionally absent.
