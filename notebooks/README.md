# Notebooks (exploratory)

These Google Colab notebooks are side experiments. Their results are **not** part of
the evaluation reported in the final report.

| Notebook | Purpose |
|---|---|
| `FYP_GPU_Colab.ipynb` | Runs `scripts/benchmark.py` on a Colab NVIDIA GPU in CPU-only, CPU + GPU and GPU set-ups, as a step towards the dedicated-GPU future work in Chapter 6. Expects a `colab_bench.zip` of this repository (plus a recorded clip in `recordings/`) in a Google Drive folder `FYP_colab`. |
| `FYP_VLM_Occupancy_Test.ipynb` | Small test of whether an open vision-language model (Qwen2.5-VL-3B-Instruct) can judge seat occupancy from still frames. Expects a `vlm_test.zip` of frames and labels in the same Drive folder. |

The zip files are not included, because they contain frames from the recorded sessions.
