The LL/BT/HT Latin-square rank traces are 250-275 MB gzipped each (the m=4096/8192
LL cells dominate) and exceed GitHub file limits, so they are not committed.
Regenerate with:
  FLASHINFER_CUTE_DSL_DISABLE_CACHE=1 run-iket --output-dir out/mnnvl --clobber \
      profile --postprocess all -- torchrun --nproc-per-node 8 scripts/mnnvl_protocols_iket_driver.py
The analysis in results/mnnvl_protocol_report.txt and logs/mnnvl_run.log derives from them.
