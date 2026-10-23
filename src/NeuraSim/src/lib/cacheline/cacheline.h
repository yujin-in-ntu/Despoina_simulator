// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
#ifndef CACHE_LINE_H
#define CACHE_LINE_H

#include <iostream>

using namespace std;

class CacheLine {
	public:
		CacheLine(	int payload,
					long long address,
					long long accessed_at_cycle);
		~CacheLine();
		int payload;
		long long address;
		long long accessed_at_cycle;
	private:
};

#endif // !CACHE_LINE_H