# File-scoped license map

No blanket license applies to this distribution.

## CC BY 4.0 — NeuraSim-derived runtime

`LICENSES/CC-BY-4.0.txt` applies to files marked
`SPDX-License-Identifier: CC-BY-4.0` in:

- `src/NeuraSim/src/code/`
- the root runtime support files `functions.*` and `config.h`
- `src/NeuraSim/src/lib/{argparse,cache,cacheline,connector,counters,debugger,dispatcher,hashline,hashpad,hashreq,hashresp,instx,mc,memreq,memresp,neuracore,neuramem,neurarouter,neuratile,writer}/`

These files derive from NeuraChip/NeuraSim, DOI
`10.5281/zenodo.10896280`. See `NOTICE`.

## Apache-2.0 — new Despoina work

`LICENSES/Apache-2.0.txt` applies to files marked
`SPDX-License-Identifier: Apache-2.0`, including:

- `src/NeuraSim/src/lib/{activation,buffer,pe,neuracim,result_sink}/`
- `src/NeuraSim/src/lib/{energy_params_generated.h,neurosim_params.h}`
- `src/NeuraSim/tests/`, root `tests/`, `scripts/`, `configs/`, and root
  documentation/build metadata

## Third party

- DRAMsim3: MIT (`third_party/DRAMsim3-1.0.0/LICENSE`)
- fmt: BSD-2-Clause (`third_party/DRAMsim3-1.0.0/ext/fmt/LICENSE.rst`)
- INIReader/inih: BSD-3-Clause
  (`third_party/DRAMsim3-1.0.0/ext/headers/INIHLICENSE.txt`)
- JSON for Modern C++: embedded MIT notice in `json.hpp`

Apache-2.0 does not replace any CC BY or third-party license.
