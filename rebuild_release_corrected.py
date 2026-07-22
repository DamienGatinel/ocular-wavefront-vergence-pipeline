#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED -- do not use.

This early driver used an older column-selection path and wrote the private
re-identification mapping inside its output directory. It has been superseded by
build_release.py, which is the single canonical, exact reproducer:

  - keeps the LAST (corrected) duplicate calc_* column, renamed to its base name;
  - keeps CLASSIFIksi in the output but excludes it from the topography argmax;
  - applies the PTVaxis correction and the post-hoc 20 D QC rule;
  - writes the 9-field dictionary and the FIVE tables (incl. qc_sensitivity.csv);
  - writes the private mapping OUTSIDE the public tree (PRIVATE_OUTPUT_DIR);
  - manifests the whole public package and asserts 1892 x 1235.

Use:  OPD_AGGREGATE_XLSX=... OPD_SALT_FILE=... python build_release.py
"""
import sys
sys.exit("rebuild_release_corrected.py is DEPRECATED; use build_release.py instead.")
