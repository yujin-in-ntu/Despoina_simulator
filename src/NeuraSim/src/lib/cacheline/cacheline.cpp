// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
// Created by Kaustubh Shivdikar
//

#include "cacheline.h"


using namespace std;

CacheLine::CacheLine(	int payload,
						long long address,
						long long accessed_at_cycle) {
	// Constructor
	this->payload = payload;
	this->address = address;
	this->accessed_at_cycle = accessed_at_cycle;
}

CacheLine::~CacheLine() {
	// Destructor
}
