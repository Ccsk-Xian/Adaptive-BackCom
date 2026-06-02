| Model | BER | Parameters | MACs |
| :--- | :---: | :---: | :---: |
| **PATD-1D (Normal)** | 4.16% | 15.6K | 10.69M |
| **PATD-1D (Tiny)** | 4.22% | 8.5K | 10.62M |
| **TCN** | 4.31% | 269.6K | 843.75M |
| **GRU** | 4.45% | 13.2K | 66.85M |
| **LSTM** | 4.56% | 357.4K | 463.6M |
| **Transformer** | 4.74% | 314.11K | 680.53M |


Notably, the codebase of PATD-1D has been adapted to ensure compatibility with STM32CubeMX. Specifically, model parameters are explicitly fixed to the maximum extent possible. Furthermore, considering that 2D convolutional operators offer superior deployment optimization compared to their 1D counterparts, we implemented the 1D convolution using a 2D structure. Crucially, its underlying mechanism remains identical to the 1D convolutional operator described in the paper
