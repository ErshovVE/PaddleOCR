# RU OCR speed benchmark

- GPU: none
- CPU: AMD EPYC 7B12, 1 socket(s) x 2 cores x 2 threads; 4 logical on the host, **4 available to the process** (affinity 4, quota 4.0)
- RAM: 31.3 GB (30.1 GB available)
- Python 3.13.15; paddlepaddle 3.3.1, paddlex 3.7.2, onnxruntime-openvino 1.24.1, opencv-python 4.14.0.94; paddle2onnx 2.0.2rc3
- Model: ru-ocr-checkpoints-ft4/best_accuracy (PP-OCRv6_small_rec); lines: all_val5k_clean_w_h.txt, 4241 lines; images decoded in memory; 3 warm-up calls not timed; library default threads

| backend | batch | sorted | lines | load_s | total_s | lines_per_s | ms_per_line | ms_per_call_mean | ms_per_call_p50 | ms_per_call_p95 | acc | acc_rstrip | same_as_paddle_cpu_b1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paddle cpu | 1 | False | 4241 | 0.92 | 617.26 | 6.9 | 145.55 | 145.54 | 151.5 | 230.25 | 0.9623 | 0.9625 | 1.0 |
| paddle cpu | 4 | False | 4241 | 0.91 | 528.01 | 8.0 | 124.5 | 497.65 | 499.79 | 677.57 | 0.963 | 0.9642 | 0.9811 |
| paddle cpu | 4 | True | 4241 | 0.92 | 344.61 | 12.3 | 81.26 | 324.79 | 322.52 | 569.47 | 0.9625 | 0.9627 | 0.9991 |
| paddle cpu | 8 | False | 4241 | 0.93 | 537.89 | 7.9 | 126.83 | 1012.97 | 968.47 | 1338.34 | 0.9611 | 0.9627 | 0.9776 |
| paddle cpu | 8 | True | 4241 | 0.94 | 310.87 | 13.6 | 73.3 | 585.44 | 591.21 | 1055.57 | 0.9637 | 0.9639 | 0.9974 |
| onnxruntime cpu | 1 | False | 4241 | 0.1 | 230.2 | 18.4 | 54.28 | 54.28 | 53.66 | 102.27 | 0.9623 | 0.9625 | 1.0 |
| onnxruntime cpu | 4 | False | 4241 | 0.1 | 326.86 | 13.0 | 77.07 | 308.06 | 306.71 | 446.97 | 0.963 | 0.9642 | 0.9811 |
| onnxruntime cpu | 4 | True | 4241 | 0.1 | 206.78 | 20.5 | 48.76 | 194.89 | 200.04 | 359.1 | 0.9625 | 0.9627 | 0.9991 |
| onnxruntime cpu | 8 | False | 4241 | 0.1 | 376.12 | 11.3 | 88.69 | 708.32 | 677.93 | 946.57 | 0.9611 | 0.9627 | 0.9776 |
| onnxruntime cpu | 8 | True | 4241 | 0.1 | 233.72 | 18.1 | 55.11 | 440.15 | 474.05 | 712.45 | 0.9637 | 0.9639 | 0.9974 |
| openvino cpu | 1 | False | 4241 | 0.31 | 229.12 | 18.5 | 54.03 | 54.02 | 54.5 | 104.34 | 0.9623 | 0.9625 | 1.0 |
| openvino cpu | 4 | False | 4241 | 0.31 | 310.59 | 13.7 | 73.24 | 292.73 | 291.42 | 417.16 | 0.963 | 0.9642 | 0.9811 |
| openvino cpu | 4 | True | 4241 | 0.31 | 198.72 | 21.3 | 46.86 | 187.29 | 188.03 | 366.6 | 0.9625 | 0.9627 | 0.9991 |
| openvino cpu | 8 | False | 4241 | 0.31 | 345.48 | 12.3 | 81.46 | 650.61 | 626.91 | 857.57 | 0.9611 | 0.9627 | 0.9776 |
| openvino cpu | 8 | True | 4241 | 0.28 | 198.22 | 21.4 | 46.74 | 373.3 | 373.31 | 703.45 | 0.9637 | 0.9639 | 0.9974 |
