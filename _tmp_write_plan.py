import pathlib
import os

BT = chr(96)
BT3 = chr(96) * 3
Q3 = chr(34) * 3  # triple double-quote

# Build plan content with proper backticks
plan_parts = []

plan_parts.append(f"""# Plan: Parallelize Stage 1 & Stage 2 Multidimensional Probing

## Overview

Parallelize the sequential parameter/block loops in {BT}probe_scales_multidim_stage1(){BT} and {BT}probe_scales_multidim_stage2_blocks(){BT} using the proven ProcessPoolExecutor pattern from CMA-ES.

**File**: {BT}modules/camera_calibration/wand_calibration/full_global_search.py{BT}
**Worker count**: {BT}max_workers = max(1, int(os.cpu_count() * 0.8)){BT} -- hardcoded, no config options.
""")