# Upstream foundations

RadarFACT-Hardware is an independent reproduction and extension built on the
open GRT ecosystem.

## Primary upstream resources

- GRT project: https://wiselabcmu.github.io/grt/
- GRT paper: https://arxiv.org/abs/2509.12482
- GRT research code: https://github.com/WiseLabCMU/grt
- Red Rover acquisition system: https://radarml.github.io/red-rover/
- NRDK implementation/tooling: https://radarml.github.io/nrdk/

The original GRT work introduced a compact multimodal acquisition platform,
I/Q-1M, an open raw-radar toolchain, and the Generalizable Radar Transformer.
Those contributions are the technical foundation of this repository.

## Scope of this repository

This repository adds an independent hardware adaptation around a Jetson-class
computer, TI xWR18xx/DCA1000 radar capture, Livox Mid-360, Hikrobot camera,
STM32 synchronization, calibration utilities, and I/Q-1M-like export tools.
It does not replace or redistribute the complete official GRT implementation.

Users should consult upstream documentation for authoritative GRT, Red Rover,
NRDK, dataset, and model instructions.

