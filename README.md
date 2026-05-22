# 1. Generating data (data.generating.py):
Change `FRAME_LEN` and `N` according to the practical BackCom system settings.
Change `pilot_bits = np.array([1, 0] * (K // 2 + 1), dtype=int)[:32]` according to the practical pilot symbol order.
Change SNR_dB = np.arange(-10, 25, 5) according to the practical SNR range considered in real scenarios.

Change Delta_beta_dB in generate_channel_coefficients_backscatter() according to the practical relative coefficient ζ.

Note: The fluctuation is currently set to 5 dB instead of 20 dB in the submitted manuscript. This inconsistency will be corrected in the next revision.

To generate Laplacian noise data, uncomment the following code:
  # Laplacian scale parameter with noise power w_b
  b = np.sqrt(w_b / 4)
  
  # Independent Laplace noise for real and imaginary parts
  n_real = rng_main.laplace(loc=0.0, scale=b, size=FRAME_LEN)
  n_imag = rng_main.laplace(loc=0.0, scale=b, size=FRAME_LEN)
  
  # Complex Laplacian noise
  l = n_real + 1j * n_imag
