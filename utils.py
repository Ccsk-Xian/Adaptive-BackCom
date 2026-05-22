import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.special import erfc
import random


def add_impulse_only_to_existing(
    frames_real: np.ndarray,
    frames_imag: np.ndarray,
    *,
    w_imp: float = 0.1,     # 额外冲击噪声的平均功率（新增噪声功率）
    p: float = 0.005,       # 冲击概率
    seed: int = 2025,       # 固定seed可复现
    dist: str = "gaussian"  # "gaussian" 或 "laplace"
):
    """
    只在概率 p 的采样点叠加冲击噪声，其余点加 0。
    保证新增冲击噪声的平均功率 E|n_imp|^2 = w_imp。

    输入 frames 已经包含 AWGN（功率=1 或其他），这里不再加背景噪声。
    """
    assert frames_real.shape == frames_imag.shape
    F, L = frames_real.shape
    rng = np.random.default_rng(seed)

    u = rng.random((F, L)) < p  # 冲击掩码

    # 让新增噪声满足：E|n_imp|^2 = p * sigma_i2 = w_imp
    # => sigma_i2 = w_imp / p
    sigma_i2 = w_imp / max(p, 1e-12)

    if dist.lower() == "gaussian":
        # 冲击点：CN(0, sigma_i2)，非冲击点：0
        ni = np.sqrt(sigma_i2 / 2.0) * (rng.standard_normal((F, L)) + 1j * rng.standard_normal((F, L)))
        n_imp = np.where(u, ni, 0.0 + 0.0j).astype(np.complex64)

    elif dist.lower() == "laplace":
        # complex Laplace: E|n|^2 = 4b^2 => b = sqrt(sigma_i2/4)
        b_i = np.sqrt(sigma_i2 / 4.0)
        ni = rng.laplace(0.0, b_i, size=(F, L)) + 1j * rng.laplace(0.0, b_i, size=(F, L))
        n_imp = np.where(u, ni, 0.0 + 0.0j).astype(np.complex64)

    else:
        raise ValueError("dist must be 'gaussian' or 'laplace'")

    y = frames_real.astype(np.float32) + 1j * frames_imag.astype(np.float32)
    y_new = y + n_imp
    return y_new.real.astype(np.float32), y_new.imag.astype(np.float32)

class EarlyStopping:
    def __init__(self, patience=5, mode='max', verbose=True, delta=0.0, path='best_model.pt'):
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.best_epoch = 0

    def __call__(self, score, model, epoch):
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            self.best_epoch = epoch
        elif  score < self.best_score + self.delta:
            # print (score)
            self.counter += 1
            if self.verbose:
                print(f" EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            self.best_epoch = epoch

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f" Model saved to {self.path}")

class EarlyStoppingWithLR:
    def __init__(self, optimizer, 
                 reduce_patience=5, 
                 stop_patience=10, 
                 mode='max', 
                 delta=0.0, 
                 verbose=True, 
                 path='best_model.pt',
                 lr_factor=0.1):
        """
        optimizer: torch.optim.XXX
        reduce_patience: 连续多少次没有提升 → 降学习率
        stop_patience: 连续多少次没有提升 → early stop
        lr_factor: 下降倍数（默认 0.1）
        """
        self.optimizer = optimizer
        self.reduce_patience = reduce_patience
        self.stop_patience = stop_patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.path = path
        
        self.counter = 0           # 统计连续“不提升”的次数
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.lr_factor = lr_factor
        self.red = 0

    def __call__(self, score, model, epoch):
        # mode = 'min' 则反向（比如 loss）
        if self.mode == 'min':
            score = -score

        # --- 第一次调用 ---
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            self.best_epoch = epoch
            return

        # --- 判断是否提升 ---
        if score < self.best_score + self.delta:
            # 没有提升
            self.counter += 1
            if self.verbose:
                print(f" No improvement: {self.counter} epochs")

            # ------ 第一阶段：降学习率 ------
            if self.red <= 2 and self.counter == self.reduce_patience:
                self.adjust_learning_rate()
                if self.verbose:
                    print(f" LR reduced ×{self.lr_factor}")
                # 重置计数器
                self.counter = 0
                self.red += 1

            # ------ 第二阶段：停止训练 ------
            if self.counter >= self.stop_patience:
                if self.verbose:
                    print(f" Early stopping at epoch {epoch}")
                self.early_stop = True

        else:
            # 出现提升，更新 best
            self.best_score = score
            self.save_checkpoint(model)
            self.best_epoch = epoch

            # 重置计数器
            self.counter = 0

    def adjust_learning_rate(self):
        """学习率缩小 factor 倍"""
        for g in self.optimizer.param_groups:
            g['lr'] *= self.lr_factor

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f" Model saved to {self.path}")

class IQDataset_cov(Dataset):
    def __init__(self, file_paths, labels, seq_len):
        self.data = []
        self.targets = []
        self.seq_len = seq_len

        for ri, label in zip(file_paths, labels):
            real = ri.real.astype(np.float32)
            imag = ri.imag.astype(np.float32)
            iq = np.stack([real, imag], axis=-1)
            # print('1111111111111')
            # print(iq.shape)
            # iq = np.fromfile(path, dtype=np.float32)
            # # print(iq[0])
            # # print(iq[1])
            # iq = iq.reshape(-1, 2)  # (I, Q)
            # max_val = np.max(np.abs(iq))
            # if max_val>0:
            #     iq = iq/max_val
            # iq = (iq - np.mean(iq, axis=0)) / (np.std(iq, axis=0) + 1e-6) 
            
            # print(iq[0])
            # print(len(iq))
            # iq = iq[:1000000]
            num_seq = len(iq) // seq_len
            # print(num_seq)
            iq = iq[:num_seq * seq_len]
            iq = iq.reshape(num_seq, seq_len, 2)  # (num_seq, seq_len, 2)
            # print(iq.shape)
            self.data.append(iq)
            self.targets.append(np.full((num_seq,), label))

        self.data = np.concatenate(self.data, axis=0)
        # print(self.data.shape)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

class IQDataset(Dataset):
    def __init__(self, file_paths, labels, seq_len):
        self.data = []
        self.targets = []
        self.seq_len = seq_len

        for ri, label in zip(file_paths, labels):
            real = ri.real.astype(np.float32)
            imag = ri.imag.astype(np.float32)
            iq = np.stack([real, imag], axis=-1)
            # print('1111111111111')
            # print(iq.shape)
            # iq = np.fromfile(path, dtype=np.float32)
            # # print(iq[0])
            # # print(iq[1])
            # iq = iq.reshape(-1, 2)  # (I, Q)
            # max_val = np.max(np.abs(iq))
            # # # db 0 4.24 5.47 
            # print(max_val)
            # # print(max_val.shape)
            # if max_val>0:
            #     iq = iq/max_val

            # max_val = np.max(np.abs(iq))
            # if max_val>0:
            #     iq = iq/max_val

            # iq = (iq - np.mean(iq, axis=0)) / (np.std(iq, axis=0) + 1e-6) 
            # div = 3.7954977
            # iq = iq/div
            # print(iq[0])
            # print(len(iq))
            # iq = iq[:1000000]
            num_seq = len(iq) // seq_len
            # print(num_seq)
            iq = iq[:num_seq * seq_len]
            iq = iq.reshape(num_seq, seq_len, 2)  # (num_seq, seq_len, 2)
            # offset = int(random.random() * 0.15 * N)
            # fn0 = fn0_d.reshape(LL, N)
            # fn1 = fn1_d.reshape(LL, N)
            # fn0[:,:offset] , fn1[:,:offset] = fn1[:,:offset] , fn0[:,:offset]  
            # print(iq.shape)

            # 对每个数据归一化
            # max_vals = np.max(np.abs(iq), axis=(1), keepdims=True)
            # # mean_vals = np.mean(np.abs(iq), axis=(1, 2), keepdims=True)
            # # iq = (iq - mean_vals) / (max_vals + 1e-8)
            # print(num_seq)
            # print(max_vals.shape)
            # iq = iq  / (max_vals + 1e-8)

            self.data.append(iq)
            self.targets.append(np.full((num_seq,), label))
        a = np.array(self.data)
        # print(a.shape)
        self.data = np.concatenate(self.data, axis=0)
        # print(self.data.shape)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


class FramePilotSymbolDatasetFrameOffsetd10(Dataset):
    """
    每个样本:
      x: shape = (num_pilot + 1, N, 2)
         前 num_pilot 个导频符号 + 当前的数据符号
         每个符号有 N 个采样点，IQ=2 通道
      y: 当前数据符号的比特标签（0 / 1）

    额外功能（模拟非精准同步）:
      - 在 __init__ 里对每一帧整体做一个循环偏移（np.roll）
      - 偏移单位是采样点，最大偏移为 ofs * N
      - 这样：
          * 最后一个导频的后半段会“侵入”第一个数据符号
          * 后面的数据符号依次错位
          * 最后一个数据符号多出来的尾巴拼接到第一个导频前端
    """

    def __init__(self,
                 frames_real_path: str,
                 frames_imag_path: str,
                 labels_path: str,
                 N: int = 70,
                 num_pilot: int = 4,
                 normalize: bool = False,
                 ofs: float = 0.2,
                 impulsive=False,
                 pattern=1,
                 db = 0):
        """
        :param ofs: 最大偏移比例 (0 ~ 1)，例如 0.15 表示最多 0.15*N 个采样点偏移
        """
        super().__init__()
        self.db = db
        self.label_smoothing = [0.29139762, 0.17893475, 0.09755288, 0.06153711, 0.04775088, 0.0420118, 0.0421346]
        self.num_classes = 2
        # 1) 加载 numpy 文件
        frames_real = np.load(frames_real_path)  # shape: (num_frames, FRAME_LEN)
        frames_imag = np.load(frames_imag_path)
        labels = np.load(labels_path)            # shape: (num_frames, K)

        self.frames_real = frames_real.astype(np.float32)
        self.frames_imag = frames_imag.astype(np.float32)
        self.labels = labels.astype(np.int64)
        if impulsive:
            self.frames_real, self.frames_imag = add_impulse_only_to_existing(
            self.frames_real, self.frames_imag,
            w_imp=0.1,     # 新增冲击噪声功率
            p=0.005,
            seed=100,
            dist="gaussian"
        )
        self.N = N
        self.num_pilot = num_pilot
        self.normalize = normalize
        self.ofs = ofs
        self.pattern = pattern

        num_frames, frame_len = self.frames_real.shape
        K = frame_len // N
        assert frame_len % N == 0, "FRAME_LEN 必须能被 N 整除"
        assert self.labels.shape == (num_frames, K)

        self.num_frames = num_frames
        self.K = K
        pilot_len = self.num_pilot * self.N  # 导频部分采样点数
        # 2) 在这里对每一帧做一次“整体时间偏移”（循环移位）
        self.frame_offsets = np.zeros(num_frames, dtype=np.int32)  # ✅新增：记录每帧偏移

        if ofs is not None and ofs > 0:
            max_offset = int(ofs * N)  # 最大偏移样本数（不跨过一个符号太多）
            if max_offset > 0:
                for i in range(num_frames):
                    # offset = random.randint(0, max_offset)
                    offset = max_offset
                    self.frame_offsets[i] = offset  # ✅新增：记录偏移
                    # # 只偏移pilots
                    # pilot_r = self.frames_real[i, :pilot_len]
                    # pilot_i = self.frames_imag[i, :pilot_len]

                    # pilot_r = np.roll(pilot_r, offset)
                    # pilot_i = np.roll(pilot_i, offset)

                    # # 拼接回原帧：导频变了，数据保持不动
                    # self.frames_real[i, :pilot_len] = pilot_r
                    # self.frames_imag[i, :pilot_len] = pilot_i

                    # # 只偏移data
                    # pilot_r = self.frames_real[i, pilot_len:]
                    # pilot_i = self.frames_imag[i, pilot_len:]

                    # pilot_r = np.roll(pilot_r, offset)
                    # pilot_i = np.roll(pilot_i, offset)

                    # # 拼接回原帧：导频变了，数据保持不动
                    # self.frames_real[i, pilot_len:] = pilot_r
                    # self.frames_imag[i, pilot_len:] = pilot_i

                    if offset > 0:
                        self.frames_real[i] = np.roll(self.frames_real[i], offset)
                        self.frames_imag[i] = np.roll(self.frames_imag[i], offset)
        else:
            # ofs=0 相当于完美同步
            self.frame_offsets[:] = 0
        # if ofs is not None and ofs > 0:
        #     max_offset = int(ofs * N)  # 最大偏移样本数（不跨过一个符号太多）
        #     if max_offset > 0:
        #         for i in range(num_frames):
        #             offset = random.randint(0, max_offset)
        #             if offset > 0:
        #                 # 正偏移：把最后 offset 个采样移到最前面
        #                 # 对应你说的“最后一个数据符号多出来的数据拼接到第一个导频前端”
        #                 self.frames_real[i] = np.roll(self.frames_real[i], offset)
        #                 self.frames_imag[i] = np.roll(self.frames_imag[i], offset)
        # 如果 ofs=0，则不做偏移，相当于完美同步

        # 3) 建立所有样本的索引 (frame_idx, sym_idx)
        # sym_idx 从 num_pilot 开始，因为前 num_pilot 是导频，不作为标签
        self.index = [
            (frame_idx, sym_idx)
            for frame_idx in range(num_frames)
            for sym_idx in range(num_pilot, K)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        frame_idx, sym_idx = self.index[idx]

        frame_r = self.frames_real[frame_idx]  # shape: (FRAME_LEN,)
        frame_i = self.frames_imag[frame_idx]

        segments = []
        offset = self.frame_offsets[frame_idx]  # 单位：采样点

        # # (1) 前 num_pilot 个导频符号（已经是在“偏移后的帧”上切）
        # for p in range(self.num_pilot):
        #     start = p * self.N
        #     end = (p + 1) * self.N
        #     seg_r = frame_r[start:end]
        #     seg_i = frame_i[start:end]
        #     seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
        #     segments.append(seg)

        # # (2) 当前的数据符号（同样是在偏移后的帧上切）
        # start = sym_idx * self.N
        # end = (sym_idx + 1) * self.N
        # seg_r = frame_r[start:end]
        # seg_i = frame_i[start:end]
        # seg_data = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
        # segments.append(seg_data)
        # (2) 当前的数据符号
        start = sym_idx * self.N
        end = (sym_idx + 1) * self.N
        seg_r = frame_r[start:end]
        seg_i = frame_i[start:end]
        seg_data = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
        
        # (1) 前 num_pilot 个导频
        if self.pattern==1:
            for p in range(self.num_pilot):
                # if p%2==0:
                #     segments.append(seg_data)
                start = p * self.N
                end = (p + 1) * self.N
                seg_r = frame_r[start:end]
                seg_i = frame_i[start:end]
                seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
                segments.append(seg)
            segments.append(seg_data)
             # 试试pilots+当前data的中间50%，这样天然不受到小的偏移影响。
            # 取当前数据符号中间50%
            # mid_start = int(0.25 * self.N)
            # mid_end   = int(0.75 * self.N)
            # seg_data_mid = seg_data[mid_start:mid_end]    # (0.5N, 2)

            # for p in range(self.num_pilot):
            #     # 取第 p 个 pilot 符号
            #     p_start = p * self.N
            #     p_end   = (p + 1) * self.N
            #     pilot_r = frame_r[p_start:p_end]
            #     pilot_i = frame_i[p_start:p_end]
            #     pilot_seg = np.stack([pilot_r, pilot_i], axis=-1)  # (N, 2)

            #     # pilot 中间50%
            #     pilot_mid = pilot_seg[mid_start:mid_end]           # (0.5N, 2)

            #     # 拼接：pilot_mid 在前，data_mid 在后 -> (N,2)
            #     combined = np.concatenate([pilot_mid, seg_data_mid], axis=0)

            #     segments.append(combined)

            # for p in range(self.num_pilot):
            #     # 取第 p 个 pilot 符号
            #     if p%2==0:
            #         segments.append(seg_data)
            #     p_start = p * self.N
            #     p_end   = (p + 1) * self.N
            #     pilot_r = frame_r[p_start:p_end]
            #     pilot_i = frame_i[p_start:p_end]
            #     pilot_seg = np.stack([pilot_r, pilot_i], axis=-1)  # (N, 2)

            #     # pilot 中间50%
            #     pilot_mid = pilot_seg[mid_start:mid_end]           # (0.5N, 2)

            #     # 拼接：pilot_mid 在前，data_mid 在后 -> (N,2)
            #     combined = np.concatenate([pilot_mid, seg_data_mid], axis=0)
            #     segments.append(pilot_seg)
            #     segments.append(combined)


            # for p in range(self.num_pilot//2):
            #     if p%2==0:
            #         segments.append(seg_data)
            #     p = p*2+1
            #     start = p * self.N
            #     end = (p + 1) * self.N
            #     seg_r = frame_r[start:end]
            #     seg_i = frame_i[start:end]
            #     seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
            #     segments.append(seg)
        elif self.pattern==2:
            for p in range(self.num_pilot//2):
                diff = self.num_pilot//2
                segments.append(seg_data)
                start = p * self.N
                end = (p + 1) * self.N
                seg_r = frame_r[start:end]
                seg_i = frame_i[start:end]
                start = (p+diff) * self.N
                end = (p+diff+1) * self.N
                seg_r_0 = frame_r[start:end]
                seg_i_0 = frame_i[start:end]
                seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
                segments.append(seg)
                seg_0 = np.stack([seg_r_0, seg_i_0], axis=-1)  # (N, 2)
                segments.append(seg_0)
        else:
            for p in range(1,self.num_pilot//2-1):
                diff = self.num_pilot//2
                segments.append(seg_data)
                start = p * self.N
                end = (p + 1) * self.N
                seg_r = frame_r[start:end]
                seg_i = frame_i[start:end]
                start = (p+diff) * self.N
                end = (p+diff+1) * self.N
                seg_r_0 = frame_r[start:end]
                seg_i_0 = frame_i[start:end]
                seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
                segments.append(seg)
                seg_0 = np.stack([seg_r_0, seg_i_0], axis=-1)  # (N, 2)
                segments.append(seg_0)

        # x.shape = (num_pilot + 1, N, 2)
        x = np.stack(segments, axis=0)

        # (3) 可选归一化（针对一个样本整体）
        if self.normalize:
            amp = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
            max_val = np.max(amp)
            if max_val > 0:
                x = x / (max_val + 1e-8)

        y_id = int(self.labels[frame_idx, sym_idx])   # 0 or 1


        eps = float(self.label_smoothing[self.db])
        # eps = eps + self.frame_offsets[frame_idx]/self.N
        C = self.num_classes

        # 平滑后的分布：正确类 1-eps，其余类 eps/(C-1)
        y = torch.full((C,), eps / (C - 1), dtype=torch.float32)
        y[y_id] = 1.0 - eps
        # 标签：当前符号的比特（0 or 1）
        # y = self.labels[frame_idx, sym_idx]

        x = torch.from_numpy(x).float()  # (num_pilot+1, N, 2)
        # y = torch.tensor(y, dtype=torch.long)

        return x, y

        # # 标签仍然是原来的 bit（同步错误只体现在观测，不改变“真值”）
        # y = self.labels[frame_idx, sym_idx]

        # x = torch.from_numpy(x).float()  # (num_pilot+1, N, 2)
        # y = torch.tensor(y, dtype=torch.long)

        # return x, y

class FramePilotSymbolDatasetFrameOffset(Dataset):
    """
    每个样本:
      x: shape = (num_pilot + 1, N, 2)
         前 num_pilot 个导频符号 + 当前的数据符号
         每个符号有 N 个采样点，IQ=2 通道
      y: 当前数据符号的比特标签（0 / 1）

    额外功能（模拟非精准同步）:
      - 在 __init__ 里对每一帧整体做一个循环偏移（np.roll）
      - 偏移单位是采样点，最大偏移为 ofs * N
      - 这样：
          * 最后一个导频的后半段会“侵入”第一个数据符号
          * 后面的数据符号依次错位
          * 最后一个数据符号多出来的尾巴拼接到第一个导频前端
    """

    def __init__(self,
                 frames_real_path: str,
                 frames_imag_path: str,
                 labels_path: str,
                 N: int = 70,
                 num_pilot: int = 4,
                 normalize: bool = False,
                 ofs: float = 0.15,
                 impulsive=False):
        """
        :param ofs: 最大偏移比例 (0 ~ 1)，例如 0.15 表示最多 0.15*N 个采样点偏移
        """
        super().__init__()

        # 1) 加载 numpy 文件
        frames_real = np.load(frames_real_path)  # shape: (num_frames, FRAME_LEN)
        frames_imag = np.load(frames_imag_path)
        labels = np.load(labels_path)            # shape: (num_frames, K)

        self.frames_real = frames_real.astype(np.float32)
        self.frames_imag = frames_imag.astype(np.float32)
        self.labels = labels.astype(np.int64)
        if impulsive:
            self.frames_real, self.frames_imag = add_impulse_only_to_existing(
            self.frames_real, self.frames_imag,
            w_imp=0.1,     # 新增冲击噪声功率
            p=0.005,
            seed=100,
            dist="gaussian"
        )
        self.N = N
        self.num_pilot = num_pilot
        self.normalize = normalize
        self.ofs = ofs

        num_frames, frame_len = self.frames_real.shape
        K = frame_len // N
        assert frame_len % N == 0, "FRAME_LEN 必须能被 N 整除"
        assert self.labels.shape == (num_frames, K)

        self.num_frames = num_frames
        self.K = K

        # 2) 在这里对每一帧做一次“整体时间偏移”（循环移位）
        self.frame_offsets = np.zeros(num_frames, dtype=np.int32)  # ✅新增：记录每帧偏移

        if ofs is not None and ofs > 0:
            max_offset = int(ofs * N)  # 最大偏移样本数（不跨过一个符号太多）
            if max_offset > 0:
                for i in range(num_frames):
                    offset = random.randint(0, max_offset)
                    self.frame_offsets[i] = offset  # ✅新增：记录偏移
                    if offset > 0:
                        self.frames_real[i] = np.roll(self.frames_real[i], offset)
                        self.frames_imag[i] = np.roll(self.frames_imag[i], offset)
        else:
            # ofs=0 相当于完美同步
            self.frame_offsets[:] = 0
        # if ofs is not None and ofs > 0:
        #     max_offset = int(ofs * N)  # 最大偏移样本数（不跨过一个符号太多）
        #     if max_offset > 0:
        #         for i in range(num_frames):
        #             offset = random.randint(0, max_offset)
        #             if offset > 0:
        #                 # 正偏移：把最后 offset 个采样移到最前面
        #                 # 对应你说的“最后一个数据符号多出来的数据拼接到第一个导频前端”
        #                 self.frames_real[i] = np.roll(self.frames_real[i], offset)
        #                 self.frames_imag[i] = np.roll(self.frames_imag[i], offset)
        # 如果 ofs=0，则不做偏移，相当于完美同步

        # 3) 建立所有样本的索引 (frame_idx, sym_idx)
        # sym_idx 从 num_pilot 开始，因为前 num_pilot 是导频，不作为标签
        self.index = [
            (frame_idx, sym_idx)
            for frame_idx in range(num_frames)
            for sym_idx in range(num_pilot, K)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        frame_idx, sym_idx = self.index[idx]
        offset = self.frame_offsets[frame_idx]  # 单位：采样点
        frame_r = self.frames_real[frame_idx]  # shape: (FRAME_LEN,)
        frame_i = self.frames_imag[frame_idx]

        segments = []

        # (1) 前 num_pilot 个导频符号（已经是在“偏移后的帧”上切）
        # for p in range(self.num_pilot):
        #     start = p * self.N
        #     end = (p + 1) * self.N
        #     seg_r = frame_r[start:end]
        #     seg_i = frame_i[start:end]
        #     seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
        #     segments.append(seg)
        for p in range(self.num_pilot//2):
                diff = self.num_pilot//2
                start = p * self.N
                end = (p + 1) * self.N
                seg_r = frame_r[start:end]
                seg_i = frame_i[start:end]
                start = (p+diff) * self.N
                end = (p+diff+1) * self.N
                seg_r_0 = frame_r[start:end]
                seg_i_0 = frame_i[start:end]
                seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
                segments.append(seg)
                seg_0 = np.stack([seg_r_0, seg_i_0], axis=-1)  # (N, 2)
                segments.append(seg_0)

        # (2) 当前的数据符号（同样是在偏移后的帧上切）
        start = sym_idx * self.N
        end = (sym_idx + 1) * self.N
        seg_r = frame_r[start:end]
        seg_i = frame_i[start:end]
        seg_data = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
        segments.append(seg_data)

        # x.shape = (num_pilot + 1, N, 2)
        x = np.stack(segments, axis=0)

        # (3) 可选归一化（针对一个样本整体）
        if self.normalize:
            amp = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
            max_val = np.max(amp)
            if max_val > 0:
                x = x / (max_val + 1e-8)

        # 标签仍然是原来的 bit（同步错误只体现在观测，不改变“真值”）
        y = self.labels[frame_idx, sym_idx]

        x = torch.from_numpy(x).float()  # (num_pilot+1, N, 2)
        y = torch.tensor(y, dtype=torch.long)
        offset = torch.tensor(offset, dtype=torch.long)
        return x, y,offset


class FramePilotPairDataset(Dataset):
    """
    每个样本:
      x: shape = (num_pilot + 2, N, 2)
         前 num_pilot 个导频符号 + 后面两个“待判决数据符号”
         每个符号有 N 个采样点，IQ=2 通道

      y: 当前这两个数据符号对应的 4 类标签:
         (b1, b2) ∈ {(1,1), (0,0), (1,0), (0,1)}
         映射为类别 {0,1,2,3}
    """

    def __init__(self,
                 frames_real_path: str,
                 frames_imag_path: str,
                 labels_path: str,
                 N: int = 150,
                 num_pilot: int = 8,     # ⭐ 8 个导频
                 normalize: bool = False):
        super().__init__()

        # 加载 numpy 文件
        frames_real = np.load(frames_real_path)  # shape: (num_frames, FRAME_LEN)
        frames_imag = np.load(frames_imag_path)
        labels = np.load(labels_path)            # shape: (num_frames, K)

        self.frames_real = frames_real.astype(np.float32)
        self.frames_imag = frames_imag.astype(np.float32)
        self.labels = labels.astype(np.int64)

        self.N = N
        self.num_pilot = num_pilot
        self.normalize = normalize

        num_frames, frame_len = self.frames_real.shape
        K = frame_len // N       # 总符号数
        assert frame_len % N == 0, "FRAME_LEN 必须能被 N 整除"
        assert self.labels.shape == (num_frames, K)

        self.num_frames = num_frames
        self.K = K

        # ===== 关键修改：用不重叠的两符号一组 =====
        # 前 num_pilot 个符号是导频 (0 ~ num_pilot-1)
        # 数据符号从下标 num_pilot 开始：
        #
        # 想要的对是：(9,10), (11,12), (13,14)... (1-based)
        # 即 0-based: (8,9), (10,11), (12,13)...
        #
        # 所以 sym_idx 取: num_pilot, num_pilot+2, num_pilot+4, ...
        # 并且保证 sym_idx+1 < K  → sym_idx <= K-2
        self.index = [
            (frame_idx, sym_idx)
            for frame_idx in range(num_frames)
            for sym_idx in range(num_pilot, K - 1, 2)  # ⭐ 步长 2，避免重叠
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        frame_idx, sym_idx = self.index[idx]

        frame_r = self.frames_real[frame_idx]  # shape: (FRAME_LEN,)
        frame_i = self.frames_imag[frame_idx]

        segments = []

        # (1) 前 num_pilot 个导频符号: 下标 0 ~ num_pilot-1
        for p in range(self.num_pilot):
            start = p * self.N
            end = (p + 1) * self.N
            seg_r = frame_r[start:end]
            seg_i = frame_i[start:end]
            seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
            segments.append(seg)

        # (2) 当前这一对数据符号: sym_idx 和 sym_idx+1
        for offset in range(2):   # 两个判决符号
            s = sym_idx + offset
            start = s * self.N
            end = (s + 1) * self.N
            seg_r = frame_r[start:end]
            seg_i = frame_i[start:end]
            seg_data = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
            segments.append(seg_data)

        # x.shape = (num_pilot + 2, N, 2)  -> (10, N, 2) 如果 num_pilot=8
        x = np.stack(segments, axis=0)

        # 可选归一化（针对一个样本整体）
        if self.normalize:
            amp = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
            max_val = np.max(amp)
            if max_val > 0:
                x = x / (max_val + 1e-8)

        # ====== 标签 y：由两个比特合成 4 类 ======
        # 对应的两个比特是 sym_idx, sym_idx+1
        b1 = int(self.labels[frame_idx, sym_idx])
        b2 = int(self.labels[frame_idx, sym_idx + 1])

        # 你指定的类别顺序: 11, 00, 10, 01
        pattern_to_class = {
            (1, 1): 0,
            (0, 0): 1,
            (1, 0): 2,
            (0, 1): 3,
        }
        y = pattern_to_class[(b1, b2)]

        x = torch.from_numpy(x).float()           # (num_pilot+2, N, 2)
        y = torch.tensor(y, dtype=torch.long)     # 0 ~ 3

        return x, y

class FramePilotSymbolDatasetd10(Dataset):
    """
    每个样本:
      x: shape = (num_pilot + 1, N, 2)
         前 num_pilot 个导频符号 + 当前的数据符号
         每个符号有 N 个采样点，IQ=2 通道
      y: 当前数据符号的比特标签（0 / 1）
    """

    def __init__(self,
                 frames_real_path: str,
                 frames_imag_path: str,
                 labels_path: str,
                 N: int = 70,
                 num_pilot: int = 4,
                 normalize: bool = False,
                 impulsive=False,
                 db = 0):
        super().__init__()
        self.label_smoothing = [0.29139762, 0.17893475, 0.09755288, 0.06153711, 0.04775088, 0.0420118, 0.0421346]
        self.num_classes = 2
        self.db = db
        # 加载 numpy 文件
        frames_real = np.load(frames_real_path)  # shape: (num_frames, FRAME_LEN)
        frames_imag = np.load(frames_imag_path)
        labels = np.load(labels_path)            # shape: (num_frames, K)

        self.frames_real = frames_real.astype(np.float32)
        self.frames_imag = frames_imag.astype(np.float32)
        self.labels = labels.astype(np.int64)
        if impulsive:
            self.frames_real, self.frames_imag = add_impulse_only_to_existing(
            self.frames_real, self.frames_imag,
            w_imp=0.1,     # 新增冲击噪声功率
            p=0.005,
            seed=100,
            dist="gaussian"
        )

        self.N = N
        self.num_pilot = num_pilot
        self.normalize = normalize

        num_frames, frame_len = self.frames_real.shape
        K = frame_len // N
        assert frame_len % N == 0, "FRAME_LEN 必须能被 N 整除"
        assert self.labels.shape == (num_frames, K)

        self.num_frames = num_frames
        self.K = K

        # 所有样本的索引 (frame_idx, sym_idx)
        # sym_idx 从 num_pilot 开始，因为前 num_pilot 是导频，不作为标签
        self.index = [
            (frame_idx, sym_idx)
            for frame_idx in range(num_frames)
            for sym_idx in range(num_pilot, K)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        frame_idx, sym_idx = self.index[idx]

        frame_r = self.frames_real[frame_idx]  # shape: (FRAME_LEN,)
        frame_i = self.frames_imag[frame_idx]

        segments = []

        # (2) 当前的数据符号
        start = sym_idx * self.N
        end = (sym_idx + 1) * self.N
        seg_r = frame_r[start:end]
        seg_i = frame_i[start:end]
        seg_data = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
        
        # 试试pilots+当前data的中间50%，这样天然不受到小的偏移影响。
        # 取当前数据符号中间50%
        # mid_start = int(0.25 * self.N)
        # mid_end   = int(0.75 * self.N)
        # seg_data_mid = seg_data[mid_start:mid_end]    # (0.5N, 2)

        # for p in range(self.num_pilot):
        #     # 取第 p 个 pilot 符号
        #     p_start = p * self.N
        #     p_end   = (p + 1) * self.N
        #     pilot_r = frame_r[p_start:p_end]
        #     pilot_i = frame_i[p_start:p_end]
        #     pilot_seg = np.stack([pilot_r, pilot_i], axis=-1)  # (N, 2)

        #     # pilot 中间50%
        #     pilot_mid = pilot_seg[mid_start:mid_end]           # (0.5N, 2)

        #     # 拼接：pilot_mid 在前，data_mid 在后 -> (N,2)
        #     combined = np.concatenate([pilot_mid, seg_data_mid], axis=0)

        #     segments.append(combined)

        # for p in range(self.num_pilot):
        #     # 取第 p 个 pilot 符号
        #     if p%2==0:
        #         segments.append(seg_data)
        #     p_start = p * self.N
        #     p_end   = (p + 1) * self.N
        #     pilot_r = frame_r[p_start:p_end]
        #     pilot_i = frame_i[p_start:p_end]
        #     pilot_seg = np.stack([pilot_r, pilot_i], axis=-1)  # (N, 2)

        #     # pilot 中间50%
        #     pilot_mid = pilot_seg[mid_start:mid_end]           # (0.5N, 2)

        #     # 拼接：pilot_mid 在前，data_mid 在后 -> (N,2)
        #     combined = np.concatenate([pilot_mid, seg_data_mid], axis=0)
        #     segments.append(pilot_seg)
        #     segments.append(combined)

        

        # aa = np.arange(self.num_pilot)
        # np.random.shuffle(aa)
        # (1) 前 num_pilot 个导频
        for p in range(self.num_pilot):
            # if p%2==0:
            #     segments.append(seg_data)
            start = p * self.N
            end = (p + 1) * self.N
            seg_r = frame_r[start:end]
            seg_i = frame_i[start:end]
            seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
            segments.append(seg)
            # if p%3==2:
            #     segments.append(seg_data)
        segments.append(seg_data)
        # 单0或1的pilots
        # for p in range(self.num_pilot//2):
        #     if p%2==0:
        #         segments.append(seg_data)
        #     p = p*2+1
        #     start = p * self.N
        #     end = (p + 1) * self.N
        #     seg_r = frame_r[start:end]
        #     seg_i = frame_i[start:end]
        #     seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
        #     segments.append(seg)
        
        

        # === 修正点：按符号分块，不展开 ===
        # x.shape = (num_pilot + 1, N, 2)
        x = np.stack(segments, axis=0)
        x = np.transpose(x, (2, 0, 1))      # (2, S, N)
        # 可选归一化（针对一个样本整体）幅度归一化 
        # if self.normalize:
        #     amp = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        #     max_val = np.max(amp)
        #     if max_val > 0:
        #         x = x / (max_val + 1e-8)

        # 可选归一化（针对一个样本整体）min-max归一化 
        if self.normalize:
            x_min = np.min(x)
            x_max = np.max(x)
            if x_max > x_min:
                x = (x - x_min) / (x_max - x_min + 1e-8)

        y_id = int(self.labels[frame_idx, sym_idx])   # 0 or 1

        eps = float(self.label_smoothing[self.db])
        C = self.num_classes

        # 平滑后的分布：正确类 1-eps，其余类 eps/(C-1)
        y = torch.full((C,), eps / (C - 1), dtype=torch.float32)
        y[y_id] = 1.0 - eps
        # 标签：当前符号的比特（0 or 1）
        # y = self.labels[frame_idx, sym_idx]

        x = torch.from_numpy(x).float()  # (num_pilot+1, N, 2)
        # y = torch.tensor(y, dtype=torch.long)

        return x, y


class FramePilotSymbolDatasetSIMO(Dataset):
    """
    多天线版本（Rx=M）

    输入数据文件假设：
      frames_real: (num_frames, M, FRAME_LEN)
      frames_imag: (num_frames, M, FRAME_LEN)
      labels:      (num_frames, K)

    每个样本:
      x:
        antenna_format="separate" -> (num_pilot+1, M, N, 2)
        antenna_format="concat"   -> (num_pilot+1, N, 2*M)
      y: 当前数据符号的标签（label smoothing 后的 soft label）
    """

    def __init__(self,
                 frames_real_path: str,
                 frames_imag_path: str,
                 labels_path: str,
                 N: int = 70,
                 num_pilot: int = 4,
                 normalize: bool = False,
                 impulsive: bool = False,
                 db: int = 0,
                 antenna_format: str = "separate",  # "separate" or "concat"
                 M: int = None):
        super().__init__()

        self.label_smoothing = [0.29139762, 0.17893475, 0.09755288, 0.06153711,
                                0.04775088, 0.0420118, 0.0421346]
        self.num_classes = 2
        self.db = db

        frames_real = np.load(frames_real_path)
        frames_imag = np.load(frames_imag_path)
        labels = np.load(labels_path)

        self.frames_real = frames_real.astype(np.float32)
        self.frames_imag = frames_imag.astype(np.float32)
        self.labels = labels.astype(np.int64)

        # 可选：加冲击噪声（如果你原函数支持多维数组，需要确认 add_impulse_only_to_existing 兼容）
        if impulsive:
            self.frames_real, self.frames_imag = add_impulse_only_to_existing(
                self.frames_real, self.frames_imag,
                w_imp=0.1,
                p=0.005,
                seed=100,
                dist="gaussian"
            )

        self.N = N
        self.num_pilot = num_pilot
        self.normalize = normalize
        self.antenna_format = antenna_format

        # ====== 维度检查 ======
        # 期望 frames_real: (num_frames, M, FRAME_LEN)
        assert self.frames_real.ndim == 3, f"frames_real should be 3D (num_frames,M,FRAME_LEN), got {self.frames_real.shape}"
        assert self.frames_imag.shape == self.frames_real.shape, "frames_imag must match frames_real shape"

        num_frames, M_infer, frame_len = self.frames_real.shape
        if M is not None:
            assert M == M_infer, f"M mismatch: file has M={M_infer}, but M={M} was passed"
        self.M = M_infer

        K = frame_len // N
        assert frame_len % N == 0, "FRAME_LEN 必须能被 N 整除"
        assert self.labels.shape == (num_frames, K), f"labels shape should be {(num_frames, K)}, got {self.labels.shape}"

        self.num_frames = num_frames
        self.K = K

        # 索引 (frame_idx, sym_idx) 从 num_pilot 开始
        self.index = [
            (frame_idx, sym_idx)
            for frame_idx in range(num_frames)
            for sym_idx in range(num_pilot, K)
        ]

    def __len__(self):
        return len(self.index)

    def _get_symbol_iq(self, frame_r, frame_i, sym_idx):
        """
        frame_r/frame_i: (M, FRAME_LEN)
        返回该符号段： (M, N, 2)
        """
        start = sym_idx * self.N
        end = (sym_idx + 1) * self.N

        seg_r = frame_r[:, start:end]  # (M, N)
        seg_i = frame_i[:, start:end]  # (M, N)

        seg = np.stack([seg_r, seg_i], axis=-1)  # (M, N, 2)
        return seg

    def __getitem__(self, idx):
        frame_idx, sym_idx = self.index[idx]

        # (M, FRAME_LEN)
        frame_r = self.frames_real[frame_idx]
        frame_i = self.frames_imag[frame_idx]

        segments = []

        # 当前数据符号 (M, N, 2)
        seg_data = self._get_symbol_iq(frame_r, frame_i, sym_idx)

        # pilots + 当前data
        # 你原来的代码里其实是 segments.append(seg_data)（重复把 data 放进去）
        # 这里按你注释的“导频符号 + 当前数据符号”的本意来写：pilots 用真实 pilot 段
        for p in range(self.num_pilot):
            seg_pilot = self._get_symbol_iq(frame_r, frame_i, p)  # (M, N, 2)
            segments.append(seg_pilot)
        segments.append(seg_data)

        # 先 stack 成 (num_pilot+1, M, N, 2)
        x = np.stack(segments, axis=0)
        # if self.num_pilot == 0:
        #     x = np.expand_dims(x, axis=0)  

        # ====== 可选：把天线维拼成通道 (num_pilot+1, N, 2*M) ======
        if self.antenna_format == "concat":
            # x: (T, M, N, 2) -> (T, N, M, 2) -> (T, N, 2*M)
            x = np.transpose(x, (0, 2, 1, 3))              # (T, N, M, 2)
            x = x.reshape(x.shape[0], x.shape[1], -1)      # (T, N, 2*M)
        elif self.antenna_format == "separate":
            # 保持 (T, M, N, 2)
            pass
        else:
            raise ValueError("antenna_format must be 'separate' or 'concat'")

        # ====== 可选归一化 ======
        # 你的原归一化是按 IQ 幅度 max 归一，这里兼容任意维度，只要最后一维是 IQ 或通道
        if self.normalize:
            # 如果是 separate: (T,M,N,2)；concat: (T,N,2M)
            # 都用 sqrt(sum(I^2+Q^2)) 的思想：对最后维做能量
            amp = np.sqrt(np.sum(x.astype(np.float32) ** 2, axis=-1))
            max_val = np.max(amp)
            if max_val > 0:
                x = x / (max_val + 1e-8)

        # ====== label smoothing（和你原来一致）=====
        y_id = int(self.labels[frame_idx, sym_idx])
        eps = float(self.label_smoothing[self.db])
        C = self.num_classes
        y = torch.full((C,), eps / (C - 1), dtype=torch.float32)
        y[y_id] = 1.0 - eps

        x = torch.from_numpy(x).float()
        return x, y

class FramePilotSymbolDataset(Dataset):
    """
    每个样本:
      x: shape = (num_pilot + 1, N, 2)
         前 num_pilot 个导频符号 + 当前的数据符号
         每个符号有 N 个采样点，IQ=2 通道
      y: 当前数据符号的比特标签（0 / 1）
    """

    def __init__(self,
                 frames_real_path: str,
                 frames_imag_path: str,
                 labels_path: str,
                 N: int = 70,
                 num_pilot: int = 4,
                 normalize: bool = False,
                 impulsive=False):
        super().__init__()

        # 加载 numpy 文件
        frames_real = np.load(frames_real_path)  # shape: (num_frames, FRAME_LEN)
        frames_imag = np.load(frames_imag_path)
        labels = np.load(labels_path)            # shape: (num_frames, K)

        self.frames_real = frames_real.astype(np.float32)
        self.frames_imag = frames_imag.astype(np.float32)
        self.labels = labels.astype(np.int64)

        if impulsive:
            self.frames_real, self.frames_imag = add_impulse_only_to_existing(
            self.frames_real, self.frames_imag,
            w_imp=0.1,     # 新增冲击噪声功率
            p=0.005,
            seed=100,
            dist="gaussian"
        )

        self.N = N
        self.num_pilot = num_pilot
        self.normalize = normalize

        num_frames, frame_len = self.frames_real.shape
        K = frame_len // N
        assert frame_len % N == 0, "FRAME_LEN 必须能被 N 整除"
        assert self.labels.shape == (num_frames, K)

        self.num_frames = num_frames
        self.K = K

        # 所有样本的索引 (frame_idx, sym_idx)
        # sym_idx 从 num_pilot 开始，因为前 num_pilot 是导频，不作为标签
        self.index = [
            (frame_idx, sym_idx)
            for frame_idx in range(num_frames)
            for sym_idx in range(num_pilot, K)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        frame_idx, sym_idx = self.index[idx]

        frame_r = self.frames_real[frame_idx]  # shape: (FRAME_LEN,)
        frame_i = self.frames_imag[frame_idx]

        segments = []

        # (1) 前 num_pilot 个导频
        for p in range(self.num_pilot):
            start = p * self.N
            end = (p + 1) * self.N
            seg_r = frame_r[start:end]
            seg_i = frame_i[start:end]
            seg = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
            segments.append(seg)

        # (2) 当前的数据符号
        start = sym_idx * self.N
        end = (sym_idx + 1) * self.N
        seg_r = frame_r[start:end]
        seg_i = frame_i[start:end]
        seg_data = np.stack([seg_r, seg_i], axis=-1)  # (N, 2)
        segments.append(seg_data)

        # === 修正点：按符号分块，不展开 ===
        # x.shape = (num_pilot + 1, N, 2)
        x = np.stack(segments, axis=0)

        # 可选归一化（针对一个样本整体）
        if self.normalize:
            amp = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
            max_val = np.max(amp)
            if max_val > 0:
                x = x / (max_val + 1e-8)

        # 标签：当前符号的比特（0 or 1）
        y = self.labels[frame_idx, sym_idx]

        x = torch.from_numpy(x).float()  # (num_pilot+1, N, 2)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

class IQDatasetwithChannel(Dataset):
    def __init__(self, file_paths, labels,channel, seq_len):
        self.data = []
        self.targets = []
        self.channel = []
        self.seq_len = seq_len

        for ri, label,chan in zip(file_paths, labels,channel):
            real = ri.real.astype(np.float32)
            imag = ri.imag.astype(np.float32)
            iq = np.stack([real, imag], axis=-1)

            # print('1111111111111')
            # print(iq.shape)
            # iq = np.fromfile(path, dtype=np.float32)
            # # print(iq[0])
            # # print(iq[1])
            # iq = iq.reshape(-1, 2)  # (I, Q)
            # max_val = np.max(np.abs(iq))
            # # # db 0 4.24 5.47 
            # print(max_val)
            # # print(max_val.shape)
            # if max_val>0:
            #     iq = iq/max_val

            # max_val = np.max(np.abs(iq))
            # if max_val>0:
            #     iq = iq/max_val

            # iq = (iq - np.mean(iq, axis=0)) / (np.std(iq, axis=0) + 1e-6) 
            # div = 3.7954977
            # iq = iq/div
            # print(iq[0])
            # print(len(iq))
            # iq = iq[:1000000]
            num_seq = len(iq) // seq_len
            # print(num_seq)
            iq = iq[:num_seq * seq_len]
            iq = iq.reshape(num_seq, seq_len, 2)  # (num_seq, seq_len, 2)
            # offset = int(random.random() * 0.15 * N)
            # fn0 = fn0_d.reshape(LL, N)
            # fn1 = fn1_d.reshape(LL, N)
            # fn0[:,:offset] , fn1[:,:offset] = fn1[:,:offset] , fn0[:,:offset]  
            # print(iq.shape)

            # 对每个数据归一化
            max_vals = np.max(np.abs(iq), axis=(1), keepdims=True)
            # mean_vals = np.mean(np.abs(iq), axis=(1, 2), keepdims=True)
            # iq = (iq - mean_vals) / (max_vals + 1e-8)
            print(num_seq)
            print(max_vals.shape)
            iq = iq  / (max_vals + 1e-8)
            channel_tensor = torch.complex(
                torch.tensor(chan.real, dtype=torch.float32),
                torch.tensor(chan.imag, dtype=torch.float32)
            )

            self.data.append(iq)
            self.targets.append(np.full((num_seq,), label))
            self.channel.append(channel_tensor)
        a = np.array(self.data)
        # print(a.shape)
        self.data = np.concatenate(self.data, axis=0)
        # print(self.data.shape)
        self.targets = np.concatenate(self.targets, axis=0)

        self.channel = np.concatenate(self.channel, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        z = self.channel[idx]
        return x, y,z

class IQDataset_s(Dataset):
    def __init__(self, file_paths, labels,m ,seq_len):
        self.data = []
        self.targets = []
        self.seq_len = seq_len

        for ri, label in zip(file_paths, labels):
            real = ri.real.astype(np.float32)
            imag = ri.imag.astype(np.float32)
            iq = np.stack([real, imag], axis=-1)
            # print('1111111111111')
            # print(iq.shape)
            # iq = np.fromfile(path, dtype=np.float32)
            # # print(iq[0])
            # # print(iq[1])
            # iq = iq.reshape(-1, 2)  # (I, Q)
            # max_val = np.max(np.abs(iq))
            # if max_val>0:
            #     iq = iq/max_val
            # iq = (iq - np.mean(iq, axis=0)) / (np.std(iq, axis=0) + 1e-6) 
            
            # print(iq[0])
            # print(len(iq))
            # iq = iq[:1000000]
            num_seq = len(iq) // seq_len
            # print(num_seq)
            iq = iq[:num_seq * seq_len]
            iq = iq.reshape(num_seq, seq_len, 2)  # (num_seq, seq_len, 2)
            # offset = int(random.random() * 0.15 * N)
            # fn0 = fn0_d.reshape(LL, N)
            # fn1 = fn1_d.reshape(LL, N)
            # fn0[:,:offset] , fn1[:,:offset] = fn1[:,:offset] , fn0[:,:offset]  
            # print(iq.shape)
            self.data.append(iq)
            self.targets.append(np.full((num_seq,), m))
        a = np.array(self.data)
        # print(a.shape)
        self.data = np.concatenate(self.data, axis=0)
        # print(self.data.shape)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

class IQDataset_offset(Dataset):
    def __init__(self, file_paths, labels, seq_len,ofs):
        self.data = []
        self.targets = []
        self.seq_len = seq_len

        for ri, label in zip(file_paths, labels):
            real = ri.real.astype(np.float32)
            imag = ri.imag.astype(np.float32)
            iq = np.stack([real, imag], axis=-1)
            # print('1111111111111')
            # print(iq.shape)
            # iq = np.fromfile(path, dtype=np.float32)
            # # print(iq[0])
            # # print(iq[1])
            # iq = iq.reshape(-1, 2)  # (I, Q)
            # 整体归一化，不太好
            max_val = np.max(np.abs(iq))
            if max_val>0:
                iq = iq/max_val
            # iq = (iq - np.mean(iq, axis=0)) / (np.std(iq, axis=0) + 1e-6) 
            
            # print(iq[0])
            # print(len(iq))
            # iq = iq[:1000000]
            num_seq = len(iq) // seq_len
            # print(num_seq)
            iq = iq[:num_seq * seq_len]
            iq = iq.reshape(num_seq, seq_len, 2)  # (num_seq, seq_len, 2)

            # 对每个数据归一化
            # max_vals = np.max(np.abs(iq), axis=(1, 2), keepdims=True)
            # iq = iq / (max_vals + 1e-8)

            # offset = int(random.random() * 0.15 * N)
            # fn0 = fn0_d.reshape(LL, N)
            # fn1 = fn1_d.reshape(LL, N)
            # fn0[:,:offset] , fn1[:,:offset] = fn1[:,:offset] , fn0[:,:offset]  
            # print(iq.shape)
            self.data.append(iq)
            self.targets.append(np.full((num_seq,), label))

        # 之前的---offset是固定的数，代码错误
        # offset = int(random.random() * ofs * seq_len)
        # offset = int(0.15*seq_len)
        # # print(offset)
        # # print(type(offset))
        # if not isinstance(self.data, np.ndarray):
        #     self.data = np.array(self.data)
        # temp_data = self.data.copy()
        # self.data[0,:,:offset,:],self.data[1,:,:offset,:] = temp_data[1,:,:offset,:],temp_data[0,:,:offset,:]


        offsets = [int(random.random() * ofs * seq_len) for _ in range(num_seq)]

        # 确保 self.data 是 ndarray 类型
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)

        temp_data = self.data.copy()

        # 对每一列应用不同的偏移量
        for i in range(num_seq):
            offset = offsets[i]
            # offset =  int(0.15 * seq_len)
            self.data[0, i, :offset, :] = temp_data[1, i, :offset, :]
            self.data[1, i, :offset, :] = temp_data[0, i, :offset, :]
        # self.data = temp_data
        # a = np.array(self.data)
        # print(a.shape)
        self.data = np.concatenate(self.data, axis=0)
        # print(self.data.shape)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

def load_iq(fname, max_samples):
    raw = np.fromfile(fname, dtype=np.float64)
    # 防止长度奇偶问题
    print(raw.shape)
    length = (len(raw) // 2) * 2
    raw = raw[:length]
    iq = raw[0::2] + 1j * raw[1::2]
    return iq[:max_samples]