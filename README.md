# 1. Generating data (data.generating.py):
Please change `FRAME_LEN` and `N` according to the practical BackCom system settings. 

Please change `pilot_bits = np.array([1, 0] * (K // 2 + 1), dtype=int)[:32]` according to the practical pilot symbol order.

Please change `SNR_dB = np.arange(-10, 25, 5)` according to the practical SNR range considered in real scenarios.

Please change `Delta_beta_dB` in `generate_channel_coefficients_backscatter()` according to the practical relative coefficient `ζ`.

Note: The fluctuation of ζ is actually 5 dB instead of 20 dB in the submitted manuscript. This inconsistency will be corrected in the next revision.

To generate Laplacian noise data, uncomment the following code:
```
  # Laplacian scale parameter with noise power w_b
  b = np.sqrt(w_b / 4)
  
  # Independent Laplace noise for real and imaginary parts
  n_real = rng_main.laplace(loc=0.0, scale=b, size=FRAME_LEN)
  n_imag = rng_main.laplace(loc=0.0, scale=b, size=FRAME_LEN)
  
  # Complex Laplacian noise
  l = n_real + 1j * n_imag
```
# 2. Two training paradigm:
  ## 1). Per-SNR-BIN (train_frame_test.py):
    Each SNR bin has a specific set of parameters--Delpoying multiple set of parameters on MCU.
  ## 2). Unifined training (train_frame_whole_IQ_SCM_train_test.py):
    SNR bins share a single set of parameters--Only delpoying one set of parameters on MCU.
# 3. Model Architectures:
  ## 1). IQ-based models(models/IQ):
    Model architectures for data symbol `N`=150, including PATD-1D, PATD-2D in `CNN_150.py`, and other tiny models like TCN, GRU, Transformer and LSTM
  ## 2). SCM-based models (models/SCM)
# 4. Imperfect conditions:
## 1) Laplace noise:
  Please refer to the context at 1. Generating data.
  ## 2) impulsive noise:
  Please change `impulsive=True` at data loading function `FramePilotSymbolDataset` and `FramePilotSymbolDatasetFrameOffsetd10` and setting at `add_impulse_only_to_existing` in utils.py.
  ## 3) Timing offset:
  Please change  `ofs` of `FramePilotSymbolDatasetFrameOffsetd10` in utils.py, which will generate a random timing offset `offset = random.randint(0, max_offset)`.


# 5. Quantization and deployment:
  Please refer to `lite.py` to make 8-bit quantization to tranmit model.pt (32 float on PC) to model.onnx (32 float for deployment) and model.tflite (8 bit for deployment).
  Utilizing STM32CubeMX or Keil MDK-ARM for compiling and uploading your models on MCUs. MCU deployment may involve various unexpected issues, and relevant online resources are still limited. If you encounter any deployment problems, please feel free to contact me at: ccsk1wsl@stu.xupt.edu.cn. I will try my best to help based on my deployment experience.

# 6. Supplement：
  In the content of our paper, we assume the RF source is complex Gaussian RF source and demonstrate the experimental results. However, if the architecture of Backcom is bistatic Backcom and the RF source exchange to sine wave or the modulation exchange to the QPSK (data.generating_sin.py, data.generating_qpsk.py), the BER performance will obtain a significiant improvent at 20db from 4.1% to 0.64% and 0.90%, respectively.
