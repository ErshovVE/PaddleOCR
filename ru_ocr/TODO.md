# TODO

[Русская версия](docs/ru/TODO.md)

## 1. Speed benchmark: Paddle, ONNX, OpenVINO on GPU and CPU

A separate notebook or script, published on its own and kept for tests: `kaggle/ru_benchmark.ipynb` (Kaggle 2× T4)
and the same code runnable locally.

- **Hardware first.** The output starts with what the numbers were measured on:
  - GPU: model, number of cards, memory, driver and CUDA version;
  - CPU: model, physical and logical cores, cores actually available to the process (affinity and cgroup quota:
    Kaggle gives fewer than the host has), RAM;
  - versions of paddlepaddle, paddlex, onnxruntime, openvino.
- **Data:** the val set (`all_val5k_clean_w_h.txt`, 4 241 lines), the ft4 model.
- **Backends:**
  - the native engine (Paddle inference) on GPU;
  - the native engine on CPU;
  - ONNX Runtime;
  - OpenVINO.
- **Per backend:**
  - batch 1 and a larger batch;
  - warm-up excluded;
  - lines/s, ms per line (mean, p50, p95), total time;
  - accuracy on the same lines, as a check that the backend reads the same.
- **Output:** a table file (`benchmark.tsv` + markdown) with the hardware header; the result goes into README / EXPERIMENT.

## 2. Triton Inference Server on ONNX and OpenVINO

- Model repository for the ONNX Runtime backend and for the OpenVINO backend:
  - `config.pbtxt` with dynamic batch and width (`[-1, 3, 48, -1]`);
  - dynamic batching.
- Pre-processing: resize to height 48 and normalize. Post-processing: CTC decode with `dict/ru_dict_ext011026.txt`. Done in an
  ensemble with a python backend, or on the client — choose one and document it.
- A client example and a `docker run` command.
- Throughput with `perf_analyzer`, next to the numbers from item 1.

## 3. Describe how the dataset is built and generated

In README (Data) and EXPERIMENT (5. Data), EN and RU:

- **Synthetic lines** (`doc-generator`): fonts and their rules, the 4 × 500k generation, how text and layouts are
  chosen, what the crops and label files look like.
- **Real lines** (`ocr_markup`): stroyinf crops cut by the text layer (`POST /run`), v2 and the modern batch
  (documents from 1992 on).
- **Assembly:**
  - `add_wh.py` and `merge_labels.py`, the common root of links;
  - train/val split: only the new source is split and appended to the existing train/val;
  - label cleaning (section 7);
  - packing for Kaggle with `pack_dataset.py --format tar`.
- Commands to rebuild the label files from scratch, and the sizes of each part.
