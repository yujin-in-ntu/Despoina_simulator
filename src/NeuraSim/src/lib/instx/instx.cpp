// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
// Created by Kaustubh Shivdikar
//

#include "instx.h"
// #include "../debugger/debugger.h"
#include "../config.h"

using namespace std;

InstX::InstX() {
}

InstX::~InstX() {
}

int InstX::get_type() {
	return (this->type);
}
