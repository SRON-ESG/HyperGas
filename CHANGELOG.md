## Version 0.2.0 (2025/04/30) Latest

### Bugfix

- Wrong COMB field ([Issue 103](https://github.com/zxdawn/HyperGas/issues/103))

### Main improvements

- Fix the PRISMA offset using manual GCPs ([Issue 22](https://github.com/zxdawn/HyperGas/issues/22)
- Memory issue of L3 data processing ([Issue 84](https://github.com/zxdawn/HyperGas/issues/84))
- TV filter with dynamic weights ([Issue 100](https://github.com/zxdawn/HyperGas/issues/100))
- Apply the MF and LMF based on emission rates automatically ([Issue 101](https://github.com/zxdawn/HyperGas/issues/101))
- Better IME calibration with new plume mask ([Issue 104](https://github.com/zxdawn/HyperGas/issues/104))

### Minor improvements

- Support plotting data with a zoom-in window ([Issue 102](https://github.com/zxdawn/HyperGas/issues/102))
- Dynamic resolution for the UTM projection ([Issue 110](https://github.com/zxdawn/HyperGas/issues/110))

## Version 0.1.0 (2025/01/17)

This is the basic version of HyperGas.

### Main functions

- Read three hyperspectral imagers' L1 data: EMIT, EnMAP, and PRISMA
- Retrieve CH4 and CO2 enhancement
- Denoise enhancement field
- Plume App: set plume markers, generate plume mask, calculate emission rate with uncertainty
- Batch processing scripts: generate L2 and L3 data, reprocess L2 and L3 data
