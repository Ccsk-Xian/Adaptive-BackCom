import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import random
from scipy.special import gammainc, gammaincc

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用支持中文的字体


# Q函数定义 (等同于MATLAB的qfunc)
def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


# def set_seed(seed=42):
#     # random.seed(seed)
#     # np.random.seed(seed)
#     rng_main = np.random.default_rng(42)    # 用在 h,miu, s, 噪声等
#     rng_bits = np.random.default_rng(123)   # 专门生成 data_bits

def set_seed():
    global rng_main, rng_bits
    rng_main = np.random.default_rng(42)    # 用在 h,miu, s, 噪声等
    rng_bits = np.random.default_rng(123)   # 专门生成 data_bits
    random.seed(42)  # 如果还要用 random.randint 做 offset 就顺便固定一下

def generate_channel_coefficients_backscatter(Delta_beta_dB=0, max_attempts=100):
    """
    Delta_beta_dB: 回散链路相对直射链路的功率比 (Pc/Ps) in dB
                   例如 -20 表示回散比直射弱 20 dB
    max_attempts: 最大尝试次数，避免无限循环
    """
    attempts = 0

    while attempts < max_attempts:
        real_h = rng_main.normal(0, 1, 1) / np.sqrt(2)
        imag_h = rng_main.normal(0, 1, 1) / np.sqrt(2)

        f_real = rng_main.normal(0, 1, 1) / np.sqrt(2)
        f_imag = rng_main.normal(0, 1, 1) / np.sqrt(2)
        g_real = rng_main.normal(0, 1, 1) / np.sqrt(2)
        g_imag = rng_main.normal(0, 1, 1) / np.sqrt(2)

        phi = 2 * np.pi * rng_main.random()
        Delta_beta_lin = 10 ** (Delta_beta_dB / 10.0)  # 线性比例
        alpha = np.sqrt(Delta_beta_lin) * np.exp(1j * phi)

        f = f_real + 1j * f_imag
        g = g_real + 1j * g_imag
        h = real_h + 1j * imag_h
        miu = h + f * g * alpha

        # 能量约束：保证 |miu| >= |h|
        energy_h = np.abs(h) ** 2
        energy_miu = np.abs(miu) ** 2

        if energy_miu >= energy_h:
            return h, miu, f, g

        attempts += 1

    # print(f"Warning: Maximum attempts ({max_attempts}) reached. Returning last generated coefficients.")
    return h, miu, f, g


def main():
    # ============== 基本参数设置 ==============
    set_seed()

    FRAME_LEN = 10500          # 每帧采样点数
    N = 150                     # 每个符号对应的采样点数（窗口长度）
    K = FRAME_LEN // N         # 每帧符号数（这里是 150）
    assert FRAME_LEN % N == 0, "FRAME_LEN 必须能被 N 整除"

    # 导频符号：前 4 个符号的比特固定为 [1, 1, 0, 0]
    # pilot_bits = np.array([1, 1, 0, 0], dtype=int)
    # pilot_bits = np.array([1, 0, 1, 0], dtype=int)
    # pilot_bits = np.array([1, 0, 1, 1,0, 0, 0, 1,1, 0, 1, 1,0, 0, 0, 1], dtype=int)
    # pilot_bits = np.array([1, 0, 1, 0,1, 0, 1, 0], dtype=int)
    # pilot_bits = np.array([1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0,1, 0, 1, 0], dtype=int)
    pilot_bits = np.array([1, 0] * (K // 2 + 1), dtype=int)[:32]
    # pilot_bits = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    # pilot_bits = np.array([1, 0, 1, 0,1, 0, 1, 0], dtype=int)
    # pilot_bits = np.array([1, 1, 0, 0,1, 0, 0, 1], dtype=int)
    num_pilot = len(pilot_bits)

    SNR_dB = np.arange(-10, 25, 5)  # [-10, -5, 0, 5, 10, 15, 20]
    w_b = 1                         # 噪声功率
    loop = int(10e3)                # 帧数（循环次数），注意太大会非常耗内存和硬盘

    Pd_sim = np.zeros(len(SNR_dB))     # 使用 th_ED 的仿真 BER
    Pd_sim_2 = np.zeros(len(SNR_dB))   # 使用 th_ED2 的仿真 BER
    Pd_sim_3 = np.zeros(len(SNR_dB))   # 使用 th_lRT 的仿真 BER
    Pd_sim_4 = np.zeros(len(SNR_dB))   # 使用 th_lRT_sin 的仿真 BER
    Pd_sim_2_off = np.zeros(len(SNR_dB))   # 使用 th_ED2 在offset下的仿真 BER
    Pd_sim_3_off = np.zeros(len(SNR_dB))   # 使用 th_lRT 在offset下的仿真 BER
    Pd_sim_4_off = np.zeros(len(SNR_dB))   # 使用 th_lRT_sin 在offset下的仿真 BER
    Pd_theo = np.zeros(len(SNR_dB))
    Pd_theo1 = np.zeros(len(SNR_dB))
    Pd_theo2 = np.zeros(len(SNR_dB))
    Pd_theo_s1 = np.zeros(len(SNR_dB))
    Pd_theo_s2 = np.zeros(len(SNR_dB))
    Pd_theo_s3 = np.zeros(len(SNR_dB))  # LRT
    for m in range(len(SNR_dB)):
        ps = 10 ** (SNR_dB[m] * 0.1) * w_b

        # 统计量
        err_ED = 0        # 使用 th_ED 的错误比特数
        err_ED2 = 0       # 使用 th_ED2 的错误比特数
        err_LRT = 0
        err_LRT_sin = 0
        total_bits = 0    # 总比特数（含导频）
        count = 0         # 用于理论值平均
        err_ED_off = 0
        err_ED2_off = 0
        err_LRT_off = 0
        err_LRT_off_sin = 0
        total_bits_off = 0
        # 用于存储该 SNR 下所有帧数据和标签
        frames_real = []   # list of (FRAME_LEN,)
        frames_imag = []   # list of (FRAME_LEN,)
        labels_list = []   # list of (K,)
        h_list = []        # 存每帧使用的 h
        miu_list = []      # 存每帧使用的 miu


        frames_real_test = []   # list of (FRAME_LEN,)
        frames_imag_test = []   # list of (FRAME_LEN,)
        labels_list_test = []   # list of (K,)
        h_list_test = []        # 存每帧使用的 h
        miu_list_test = []      # 存每帧使用的 miu

        Pd_theo1[m] = 0
        Pd_theo2[m] = 0

        for kkk in range(loop):
            # ====== 每个 loop 生成一个帧 ======
            
            h, miu, _, _ = generate_channel_coefficients_backscatter(Delta_beta_dB=0)
            if kkk<int(loop*0.8):
                h_list.append(h)
                miu_list.append(miu)
            else:
                h_list_test.append(h)
                miu_list_test.append(miu)
            
            sigma_0 = (np.abs(h) ** 2 * (10 ** (SNR_dB[m] * 0.1)) + w_b)
            sigma_1 = (np.abs(miu) ** 2 * (10 ** (SNR_dB[m] * 0.1)) + w_b)

            E0 = N * sigma_0
            E1 = N * sigma_1
            D0 = N * sigma_0 ** 2
            D1 = N * sigma_1 ** 2
            
            # 能量检测门限
            th_ED = 2 * N * sigma_0 * sigma_1 / (sigma_0 + sigma_1)
            th_ED2 = E0 * E1 / (E0 + E1) * (
                1 + np.sqrt(1 + (2 * (E0 + E1)) / (N * (E1 - E0)) * np.log(E1 / E0))
            )
            Th_LRT = N * sigma_0 * sigma_1 / (sigma_1 - sigma_0) * np.log(sigma_1 / sigma_0)
            n = np.arange(FRAME_LEN)
            # 生成一帧的发射信号和噪声
            w = (rng_main.standard_normal(FRAME_LEN) + 1j * rng_main.standard_normal(FRAME_LEN))
            # s = (rng_main.standard_normal(FRAME_LEN) + 1j * rng_main.standard_normal(FRAME_LEN))
            # x = np.sqrt(ps / 2) * s

            cycles_per_sym = 0        # 部署期望：每个 symbol 内约 3.2 个周期（你自己改）
            jitter_pct = 0.00           # ±2% 抖动（你自己改：0.01/0.02/0.05）

            # nominal 归一化频率：每个采样点的“周期数”
            f0_nominal = cycles_per_sym / N

            # 小范围随机：围绕 nominal 抖动
            delta = jitter_pct * f0_nominal
            f0 = rng_main.uniform(f0_nominal - delta, f0_nominal + delta)

            # 每帧初相位（可固定也可随机；建议随机）
            phi0 = rng_main.uniform(0, 2*np.pi)

            s = np.exp(1j * (2*np.pi * f0 * n + phi0))   # 幅度恒为1的复正弦
            x = np.sqrt(ps) * s                          # 幅度由 ps 控制
            # f0_norm = 0.02
            # phi0 = 2*np.pi*rng_main.random()
            

            # s = np.exp(1j * (2*np.pi*f0_norm*n + phi0))
            # x = np.sqrt(ps) * s

            l = np.sqrt(w_b / 2) * w

            b = np.sqrt(w_b / 4)   # 拉普拉斯 scale 参数，使噪声功率为 w_b

            # 实部 & 虚部独立 Laplace(0, b)
            n_real = rng_main.laplace(loc=0.0, scale=b, size=FRAME_LEN)
            n_imag = rng_main.laplace(loc=0.0, scale=b, size=FRAME_LEN)

            # l = n_real + 1j * n_imag     # (FRAME_LEN,) 复拉普拉斯噪声
           

            # 本帧的比特序列（含导频）
            data_bits = rng_bits.integers(0, 2, size=K - num_pilot)
            # random bits
            values = np.arange(0.1, 1.0, 0.1)  # [0.1, 0.2, ..., 0.9]
            random_value = rng_bits.choice(values)
            data_bits_random = rng_bits.choice([0, 1], size=K - num_pilot, p=[random_value, 1-random_value])
            data_bits = data_bits
            bits = np.concatenate([pilot_bits, data_bits])
            assert len(bits) == K

            # 本帧接收信号
            y_full = np.zeros(FRAME_LEN, dtype=np.complex128)

            # ====== 符号级发送 + 接收 + 判决 ======
            for sym_idx in range(K):
                start = sym_idx * N
                end = (sym_idx + 1) * N

                b = bits[sym_idx]
                x_seg = x[start:end]
                l_seg = l[start:end]

                if b == 0:
                    y_seg = h * x_seg + l_seg
                else:
                    y_seg = miu * x_seg + l_seg

                y_full[start:end] = y_seg


                if sym_idx <num_pilot:
                    continue
                
                

                

                if kkk>=int(loop*0.8):
                    # 能量检测
                    T = np.sum(np.abs(y_seg) ** 2)
                    d = (miu - h)                      # 复数
                    T_lrt = np.real(np.conj(d) * np.vdot(x_seg, y_seg))   # Re{ (miu-h)^* x^H y }
                    gamma = 0.5 * (np.abs(miu)**2 - np.abs(h)**2) * np.vdot(x_seg, x_seg).real
                    # 判决：T >= 阈值 => 判 1，否则判 0
                    b_hat_LRT_sin = 1 if T_lrt >= gamma else 0
                    b_hat_ED = 1 if T >= th_ED else 0
                    b_hat_ED2 = 1 if T >= th_ED2 else 0
                    b_hat_LRT = 1 if T >= Th_LRT else 0

                    if b_hat_ED != b:
                        err_ED += 1
                    if b_hat_ED2 != b:
                        err_ED2 += 1
                    if b_hat_LRT != b:
                        err_LRT += 1
                    if b_hat_LRT_sin != b:
                        err_LRT_sin += 1

                    total_bits += 1

            ofs = 0.15
            max_offset = int(ofs * N)  # 最大偏移样本数（不跨过一个符号太多）
            y_full_offset = y_full.copy()
            if max_offset > 0:
                # offset = random.randint(0, max_offset)
                offset = rng_main.integers(0, max_offset + 1)
                if offset > 0:
                    # 对整个复数帧做循环移位
                    y_full_offset = np.roll(y_full_offset, offset)

            

            for sym_idx in range(K):
                start = sym_idx * N
                end   = (sym_idx + 1) * N

                # 从偏移后的序列中截取当前符号
                y_seg_off = y_full_offset[start:end]

                # 前 4 个符号仍然跳过（导频等）
                if sym_idx < num_pilot:
                    continue

                # 这次的真实比特仍然用原来的 bits[sym_idx]
                b = bits[sym_idx]
                if kkk>=int(loop*0.8):
                    # 能量检测
                    T_off = np.sum(np.abs(y_seg_off) ** 2)
                    d = (miu - h)                      # 复数
                    T_lrt = np.real(np.conj(d) * np.vdot(x_seg, y_seg))   # Re{ (miu-h)^* x^H y }
                    gamma = 0.5 * (np.abs(miu)**2 - np.abs(h)**2) * np.vdot(x_seg, x_seg).real
                    # 判决：T >= 阈值 => 判 1，否则判 0
                    b_hat_LRT_off_sin = 1 if T_lrt >= gamma else 0
                    # 判决
                    b_hat_ED_off  = 1 if T_off >= th_ED   else 0
                    b_hat_ED2_off = 1 if T_off >= th_ED2  else 0
                    b_hat_LRT_off = 1 if T_off >= Th_LRT  else 0

                    # 统计误比特数
                    if b_hat_ED_off  != b:
                        err_ED_off  += 1
                    if b_hat_ED2_off != b:
                        err_ED2_off += 1
                    if b_hat_LRT_off != b:
                        err_LRT_off += 1
                    if b_hat_LRT_off_sin != b:
                        err_LRT_off_sin += 1

                    total_bits_off += 1

            # 保存本帧数据和标签
            if kkk>=int(loop*0.8):
                frames_real_test.append(y_full.real.astype(np.float64))
                frames_imag_test.append(y_full.imag.astype(np.float64))
                labels_list_test.append(bits.astype(np.int8))
            else:
                frames_real.append(y_full.real.astype(np.float64))
                frames_imag.append(y_full.imag.astype(np.float64))
                labels_list.append(bits.astype(np.int8))
            if kkk>=int(loop*0.8):
                # 理论 Pd 计算（按单符号）
                uu0 = (th_ED2 - E0) / np.sqrt(D0)   # 发0判1概率对应的 Q 入参
                uu1 = (E1 - th_ED2) / np.sqrt(D1)   # 发1判0概率对应的 Q 入参

                Pd_theo1[m] += 0.5 * qfunc(uu0)
                Pd_theo2[m] += 0.5 * qfunc(uu1)
                Pb_asym = qfunc(
                np.sqrt(N) * np.abs(sigma_1 - sigma_0) / (sigma_0 + sigma_1)
            )
                Pd_theo_s1[m] += Pb_asym
                gamma = ps / w_b                             # = 10**(SNR_dB/10)
                Delta = np.abs(np.abs(h)**2 - np.abs(miu)**2)
                Sigma = np.abs(h)**2 + np.abs(miu)**2

                Pb_asym2 = qfunc(
                    np.sqrt(N) * Delta / (Sigma + 2.0 / gamma)
                )
                Pd_theo_s2[m] += Pb_asym2

                sigma_max = max(sigma_0, sigma_1)
                sigma_min = min(sigma_0, sigma_1)

                # 这里假设 sigma_1 > sigma_0，如果不放心可以用 abs(...) 或者按 max/min 写
                den = (sigma_1 - sigma_0)
                
                a_min = N * (sigma_min) / den * np.log(sigma_1 / sigma_0)
                a_max = N * (sigma_max) / den * np.log(sigma_1 / sigma_0)

                # SciPy 的 gammainc / gammaincc 已经除以 Γ(N) 了，所以外面只要乘 1/2
                Pb_ML = 0.5 * (gammainc(N, a_min) + gammaincc(N, a_max))
                Pd_theo_s3[m] += Pb_ML
                count += 1

        # ====== 该 SNR 点的 BER 和理论值 ======
        Pd_theo[m] = Pd_theo1[m] / count + Pd_theo2[m] / count
        Pd_theo_s1[m] /= count
        Pd_theo_s2[m] /= count
        Pd_theo_s3[m] /= count
        Pd_sim[m] = err_ED / total_bits
        Pd_sim_2[m] = err_ED2 / total_bits
        Pd_sim_3[m] = err_LRT / total_bits
        Pd_sim_4[m] = err_LRT_sin / total_bits
        Pd_sim_2_off[m] = err_ED2_off / total_bits_off
        Pd_sim_3_off[m] = err_LRT_off / total_bits_off
        Pd_sim_4_off[m] = err_LRT_off_sin / total_bits_off

        # ====== 将本 SNR 下所有帧数据和 label 保存到文件 ======
        frames_real_arr = np.array(frames_real, dtype=np.float64)   # (loop, FRAME_LEN)
        frames_imag_arr = np.array(frames_imag, dtype=np.float64)   # (loop, FRAME_LEN)
        labels_arr = np.array(labels_list, dtype=np.int8)           # (loop, K)

        h_arr = np.array(h_list, dtype=np.complex128)
        miu_arr = np.array(miu_list, dtype=np.complex128)

        frames_real_arr_test = np.array(frames_real_test, dtype=np.float64)   # (loop, FRAME_LEN)
        frames_imag_arr_test = np.array(frames_imag_test, dtype=np.float64)   # (loop, FRAME_LEN)
        labels_arr_test = np.array(labels_list_test, dtype=np.int8)           # (loop, K)

        h_arr_test = np.array(h_list_test, dtype=np.complex128)
        miu_arr_test = np.array(miu_list_test, dtype=np.complex128)

        prefix = f'/root/signal/data/block/{SNR_dB[m]}'

        # 保存为 .npy，方便后续直接 np.load
        # _frames_real_block_frame sinr=0的时候
        # _frames_real_block_frame_f20 sinr=-20
        # np.save(prefix + '_frames_real_block_frame_150_1010_laplace_training_random.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_laplace_training_random.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_laplace_training_random.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_laplace_training_random.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_laplace_training_random.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_laplace_test_random.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_laplace_test_random.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_laplace_test_random.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_laplace_test_random.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_laplace_test_random.npy', miu_arr_test)
        
        # np.save(prefix + '_frames_real_block_frame_150_1010_laplace_training.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_laplace_training.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_laplace_training.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_laplace_training.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_laplace_training.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_laplace_test.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_laplace_test.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_laplace_test.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_laplace_test.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_laplace_test.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_16.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_16.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_16.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_16.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_16.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_16.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_16.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_16.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_16.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_16.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_16_2.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_16_2.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_16_2.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_16_2.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_16_2.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_16_2.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_16_2.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_16_2.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_16_2.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_16_2.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_8.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_8.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_8.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_8.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_8.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_8.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_8.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_8.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_8.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_8.npy', miu_arr_test)


        

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_32.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_32.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_32.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_32.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_32.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_32.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_32.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_32.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_32.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_32.npy', miu_arr_test)
        

        # sigma大小随机
        # np.save(prefix + '_frames_real_block_frame_150_1010_training_32_2.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_32_2.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_32_2.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_32_2.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_32_2.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_32_2.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_32_2.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_32_2.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_32_2.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_32_2.npy', miu_arr_test)

        # N 150
        # np.save(prefix + '_frames_real_block_frame_150_1010_training_32_sin_3.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_32_sin_3.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_32_sin_3.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_32_sin_3.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_32_sin_3.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_32_sin_3.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_32_sin_3.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_32_sin_3.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_32_sin_3.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_32_sin_3.npy', miu_arr_test)

        # N变150
        # np.save(prefix + '_frames_real_block_frame_150_1010_training_32_sin_1.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_32_sin_1.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_32_sin_1.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_32_sin_1.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_32_sin_1.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_32_sin_1.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_32_sin_1.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_32_sin_1.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_32_sin_1.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_32_sin_1.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_32_sin_9.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_32_sin_9.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_32_sin_9.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_32_sin_9.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_32_sin_9.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_32_sin_9.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_32_sin_9.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_32_sin_9.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_32_sin_9.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_32_sin_9.npy', miu_arr_test)

        np.save(prefix + '_frames_real_block_frame_150_1010_training_32_sin_0.npy', frames_real_arr)
        np.save(prefix + '_frames_imag_block_frame_150_1010_training_32_sin_0.npy', frames_imag_arr)
        np.save(prefix + '_labels_block_frame_150_1010_training_32_sin_0.npy', labels_arr)
        np.save(prefix + '_h_save_frame_150_1010_training_32_sin_0.npy', h_arr)
        np.save(prefix + '_miu_save_frame_150_1010_training_32_sin_0.npy', miu_arr)

        np.save(prefix + '_frames_real_block_frame_150_1010_test_32_sin_0.npy', frames_real_arr_test)
        np.save(prefix + '_frames_imag_block_frame_150_1010_test_32_sin_0.npy', frames_imag_arr_test)
        np.save(prefix + '_labels_block_frame_150_1010_test_32_sin_0.npy', labels_arr_test)
        np.save(prefix + '_h_save_frame_150_1010_test_32_sin_0.npy', h_arr_test)
        np.save(prefix + '_miu_save_frame_150_1010_test_32_sin_0.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_32_1100.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_32_1100.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_32_1100.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_32_1100.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_32_1100.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_32_1100.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_32_1100.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_32_1100.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_32_1100.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_32_1100.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_24.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_24.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_24.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_24.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_24.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_24.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_24.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_24.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_24.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_24.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_all.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_all.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_all.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_all.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_all.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_all.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_all.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_all.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_all.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_all.npy', miu_arr_test)

        # np.save(prefix + '_frames_real_block_frame_150_1010_training_random.npy', frames_real_arr)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_training_random.npy', frames_imag_arr)
        # np.save(prefix + '_labels_block_frame_150_1010_training_random.npy', labels_arr)
        # np.save(prefix + '_h_save_frame_150_1010_training_random.npy', h_arr)
        # np.save(prefix + '_miu_save_frame_150_1010_training_random.npy', miu_arr)

        # np.save(prefix + '_frames_real_block_frame_150_1010_test_random.npy', frames_real_arr_test)
        # np.save(prefix + '_frames_imag_block_frame_150_1010_test_random.npy', frames_imag_arr_test)
        # np.save(prefix + '_labels_block_frame_150_1010_test_random.npy', labels_arr_test)
        # np.save(prefix + '_h_save_frame_150_1010_test_random.npy', h_arr_test)
        # np.save(prefix + '_miu_save_frame_150_1010_test_random.npy', miu_arr_test)

        print(f"SNR={SNR_dB[m]} dB: 保存训练 {frames_real_arr.shape[0]} 帧, "
              f"每帧 {frames_real_arr.shape[1]} 采样点, label 维度 {labels_arr.shape}")
        print(f"SNR={SNR_dB[m]} dB: 保存测试 {frames_real_arr_test.shape[0]} 帧, "
              f"每帧 {frames_real_arr_test.shape[1]} 采样点, label 维度 {labels_arr_test.shape}")

    # ============== 画 BER 曲线 ==============
    plt.plot(SNR_dB, Pd_sim, 'o-', label='Simulation (th_ED_largeN)')
    plt.plot(SNR_dB, Pd_sim_2, '*-', label='Simulation (th_ED)')
    plt.plot(SNR_dB, Pd_sim_3, 'r-', label='Simulation (th_LRT)')
    plt.plot(SNR_dB, Pd_sim_2, 'k-', label='Simulation_offset (th_ED)')
    plt.plot(SNR_dB, Pd_sim_3, 'D-', label='Simulation_offset (th_LRT)')
    plt.plot(SNR_dB, Pd_theo, '--', label='Theoretical_energy')
    plt.plot(SNR_dB, Pd_theo_s1, 'h--', label='Theoretical1_energy_largeN')
    plt.plot(SNR_dB, Pd_theo_s2, 'p--', label='Theoretical2_energy_largeN_2')
    plt.plot(SNR_dB, Pd_theo_s3, '^--', label='Theoretical3_LRT')

    plt.yscale('log')

    # 根据您的数据范围设置合适的刻度
    yticks_positions = [  0.05,0.1, 0.4]  # 10⁻³, 10⁻², 10⁻¹, 10⁰
    yticks_labels = [  '5x10$^{-2}$', '10$^{-1}$', '4x10$^{-1}$']

    plt.yticks(yticks_positions, yticks_labels)

    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR (Perfect Sync, Frame-based)')
    plt.grid(True)
    plt.savefig('z23_frame.png')
    plt.show()

    print("Pd_sim:", Pd_sim)
    print("energy_mc:", Pd_sim_2)
    print("LRT_mc:", Pd_sim_3)
    print("LRT_sin_mc:", Pd_sim_4)
    print("energy_training_csi_mc:", Pd_sim_2_off)
    print("lrt_training_csi_mc:", Pd_sim_3_off)
    print("lrt_sin_offset:", Pd_sim_4_off)
    print("energy_theory:", Pd_theo)
    print("Pd_theo1:", Pd_theo_s1)
    print("Pd_theo2:", Pd_theo_s2)
    print("LRT_theory:", Pd_theo_s3)


if __name__ == "__main__":
    main()