



# 旧有方法兼容泛化性测试
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import  os
import torch.optim as optim
import argparse
from models import model_registry
from utils import EarlyStoppingWithLR, IQDatasetwithChannel,FramePilotSymbolDatasetd10,FramePilotPairDataset, qfunc, load_iq
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import matplotlib
import torch.nn.functional as F


# 给软标签用的loss
def soft_cross_entropy(logits, target_probs):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()

def set_seed(seed=42):
    """
    设置所有随机源的 seed，并创建全局 rng_main / rng_bits。
    """
    global rng_main, rng_bits

    # Python 内置随机
    random.seed(seed)
    
    # NumPy 随机
    np.random.seed(seed)

    # pytorch CPU 随机
    torch.manual_seed(seed)

    # pytorch CUDA 随机（如果有 GPU）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn 确保可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 关键：你的主随机流和 bit 随机流
    rng_main = np.random.default_rng(seed)        # 用于信道/噪声/s/offset 等
    rng_bits = np.random.default_rng(seed + 999)  # 用于 data_bits，不要跟主流冲突

    print(f"[INFO] Random seed set to {seed}")

def worker_init_fn(worker_id):
    """
    让 DataLoader 的每个 worker 都使用确定性随机种子。
    """
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def scm_process(x):
    B, S, N, _ = x.shape

    # 1) 组复数: (B, S, N)
    xr = x[..., 0]      # (B, S, N)
    xi = x[..., 1]      # (B, S, N)
    x_complex = torch.complex(xr, xi)

    # 2) 构造样本协方差矩阵 R_x = 1/N * X X^H
    #    这里把 S 当成 “M 个天线 / 通道”
    #    X: (B, M, N)
    X = x_complex                             # (B, M, N)
    R = (X @ X.conj().transpose(1, 2)) / N    # (B, M, M)

    # 3) 拆成实部/虚部两个通道，作为 CMNet 的输入
    R_real = R.real.unsqueeze(1)              # (B, 1, M, M)
    R_imag = R.imag.unsqueeze(1)              # (B, 1, M, M)
    # x = torch.cat([R_real, R_imag], dim=1)  # (B, 2, M, M)
    # x = torch.cat([R_real[:,:,:,:], R_imag[:,:,:,:]], dim=1)  # (B, 2, M, M)
    x = torch.cat([R_real[:,:,0,0], R_imag[:,:,0,0]], dim=1)  # (B, 2, M, M)
    return x


def main(args):
    set_seed()
    prefix_s = '/root/signal/block/150/'
    N = 150
    # h = -0.213996343204968 + 0.968842186789723j  # 0.02时，0.05同
    # miu = 1.019093937769438 + 0.809222205013335j  # 0.02时
    w_b = 1  # 噪声方差
    SCM =False
    
    # optimizer = optim.SGD(model.parameters(),
    #                           lr=LR,
    #                           momentum=0.9,
    #                           weight_decay=5e-4)

    import collections
    # print("训练集标签分布:", collections.Counter([int(i[1]) for i in train_ds]))
    # print("测试集标签分布:", collections.Counter([int(i[1]) for i in test_ds]))

    
    SNR_dB = np.arange(-10, 25, 5)
    Pd_theo1 = np.zeros(len(SNR_dB))
    Pd_theo2 = np.zeros(len(SNR_dB))
    Pd_sim = np.zeros(len(SNR_dB))
    Pd_sim_2 = np.zeros(len(SNR_dB))
    Pd_theo = np.zeros(len(SNR_dB))
    Pd_modelIQ = np.zeros((len(SNR_dB), len(SNR_dB)))
    Pd_modelSCM = np.zeros((len(SNR_dB), len(SNR_dB)))
    test_data = {}
    for m in range(len(SNR_dB)):
        
        prefix = f'/root/signal/data/block/{SNR_dB[m]}'
        frames_real_path = prefix + "_frames_real_block_frame_150_1010_training_32.npy"
        frames_imag_path = prefix + "_frames_imag_block_frame_150_1010_training_32.npy"
        labels_path      = prefix + "_labels_block_frame_150_1010_training_32.npy"

        train_ds = FramePilotSymbolDatasetd10(
            frames_real_path,
            frames_imag_path,
            labels_path,
            N=150,
            num_pilot=32,
            normalize=False,
            impulsive=False,
            db = m
                    )

        frames_real_path_test = prefix + "_frames_real_block_frame_150_1010_test_32.npy"
        frames_imag_path_test = prefix + "_frames_imag_block_frame_150_1010_test_32.npy"
        labels_path_test      = prefix + "_labels_block_frame_150_1010_test_32.npy"

        test_ds = FramePilotSymbolDatasetd10(
            frames_real_path_test,
            frames_imag_path_test,
            labels_path_test,
            N=150,
            num_pilot=32,
            normalize=False,
            impulsive=False,
            db = m
        )
        # from torch.utils.data import Subset
        # test_ds = Subset(test_ds, range(1000))
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=2,worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,worker_init_fn=worker_init_fn)
        test_data[f'{m}'] = test_loader


        DEVICE = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
        model1 = model_registry.get_model_class(args.model1)
        model1 = model1().to(DEVICE)
        if SCM:
            model2 = model_registry.get_model_class(args.model2)
            model2 = model2().to(DEVICE)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)
            early_stopper2 = EarlyStoppingWithLR(
                optimizer=optimizer2,
                reduce_patience=5,    # 连续 5 次不提升 → 降学习率
                stop_patience=10,     # 连续 10 次不提升 → 停止训练
                mode='max',
                path=prefix_s+f'weight/{m}{args.model2}'+'6_4_1.pt'
            )
        criterion = nn.CrossEntropyLoss()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
    
        # early_stopper = EarlyStoppingWithLR(patience=10, mode='max', path=prefix_s+f'weight/{args.model}'+'pure_best_model.pt')
        early_stopper1 = EarlyStoppingWithLR(
        optimizer=optimizer1,
        reduce_patience=5,    # 连续 5 次不提升 → 降学习率
        stop_patience=10,     # 连续 10 次不提升 → 停止训练
        mode='max',
        path=prefix_s+f'weight/{m}{args.model1}'+'6_4_1.pt'
    )
    

    
    # test_loader = DataLoader(test_ds, batch_size=args.batch_size,num_workers=8,pin_memory=True,persistent_workers=True,worker_init_fn=worker_init_fn)
        for epoch in range(args.epochs):
            model1.train()
            if SCM:
                model2.train()
                total_loss2 = 0
                correct2 = 0
            total_loss1 = 0
            
            correct1 = 0
            
            total = 0
            for i,(x, y) in enumerate(train_loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                # print(x.shape)
                logits1 = model1(x)
                
                preds1 = torch.argmax(logits1, dim=1)
                # correct1 += (preds1 == y).sum().item()
                correct1 += (preds1 == y.argmax(dim=1)).sum().item()
                if SCM:
                    x_cov = scm_process(x)
                    # logits2 = model2(x_cov,x)
                    logits2 = model2(x_cov)
                    preds2 = torch.argmax(logits2, dim=1)
                    correct2 += (preds2 == y.argmax(dim=1)).sum().item()
                total += y.size(0)
                
                # print("x.shape:", x.shape)
                # print("x example:", x[0][:5])
                # if i == 0:
                    # print("初始输出 logits:", logits[:5].cpu().detach().numpy())
                    # print("预测类别 preds:", preds[:10].cpu().numpy())
                    # print("真实标签 y:", y[:10].cpu().numpy())
                # loss1 = criterion(logits1, y)
                loss1 = soft_cross_entropy(logits1,y)
                
                optimizer1.zero_grad()
                
                loss1.backward()
                optimizer1.step()
                
                
                total_loss1 += loss1.item()
                
                if SCM:
                    # loss2 = criterion(logits2, y)
                    loss2 = soft_cross_entropy(logits2,y)
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()
                    total_loss2 += loss2.item()
            acc1 = correct1 / total
            print(acc1)
            if SCM:
                acc2 = correct2 / total
                print(acc2)
            
            
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss1/len(train_loader):.4f}")

            model1.eval()
            if SCM:
                model2.eval()
                correct2 = 0
            correct1 = 0
            
            total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    
                    logits1 = model1(x)
                    preds1 = torch.argmax(logits1, dim=1)
                    if SCM:

                        x_cov = scm_process(x)
                    # logits2 = model2(x_cov,x)
                        logits2 = model2(x_cov)
                        preds2 = torch.argmax(logits2, dim=1)
                        correct2 += (preds2 == y.argmax(dim=1)).sum().item()
                    # correct1 += (preds1 == y).sum().item()
                    correct1 += (preds1 == y.argmax(dim=1)).sum().item()
                    total += y.size(0)
            val_acc1 = correct1 / total
            print(val_acc1)
            if SCM:
                val_acc2 = correct2 / total
                print(val_acc2)
                early_stopper2(val_acc2, model2, epoch)
            
            
            early_stopper1(val_acc1, model1, epoch)
            if SCM:
                if early_stopper1.early_stop and early_stopper2.early_stop:
                    print(f" Early stopping at epoch {epoch+1}, best epoch was {early_stopper1.best_epoch+1} for model1 and {early_stopper2.best_epoch+1} for model2 ")
                    break
            else:
                if early_stopper1.early_stop:
                    print(f" Early stopping at epoch {epoch+1}, best epoch was {early_stopper1.best_epoch+1} for model1  ")
                    break


    

    
    

    for m in range(len(SNR_dB)):
        DEVICE = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
        model1 = model_registry.get_model_class(args.model1)
        model1 = model1().to(DEVICE)
        if SCM:
            model2 = model_registry.get_model_class(args.model2)
            model2 = model2().to(DEVICE)
            model2.load_state_dict(torch.load(prefix_s+f'weight/{m}{args.model2}'+'6_4_1.pt'))
            model2.eval()
        ### 模型评估 ###
        model1.load_state_dict(torch.load(prefix_s+f'weight/{m}{args.model1}'+'6_4_1.pt'))
        model1.eval()

        
        for mm in range(len(SNR_dB)):
            correct1 = 0
            correct2 = 0
            total = 0
            with torch.no_grad():
                for x, y in test_data[f'{mm}']:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits1 = model1(x)
                    preds1 = torch.argmax(logits1, dim=1)
                    # correct1 += (preds1 == y).sum().item()
                    correct1 += (preds1 == y.argmax(dim=1)).sum().item()
                    if SCM:
                        x_cov = scm_process(x)
                        # logits2 = model2(x_cov,x)
                        logits2 = model2(x_cov)
                        preds2 = torch.argmax(logits2, dim=1)
                        correct2 += (preds2 == y.argmax(dim=1)).sum().item()

                    total += y.size(0)

            Pd_modelIQ[m][mm] = 1- (correct1 / total)
            if SCM:
                Pd_modelSCM[m][mm] = 1- (correct2 / total)

       

    # 完美csi已知

#     Pd_theo = [0.28560606, 0.17462727, 0.09820606, 0.06198788, 0.0507697,  0.04386061,
# 0.04158788]

# 1010e3
    # Pd_sim = [0.28560606, 0.17462727, 0.09820606, 0.06198788 ,0.0507697 , 0.04386061,
#  0.04158788]

# 1100e3
    # Pd_sim = [0.28718333, 0.17641364, 0.09956667, 0.06303485, 0.05176212, 0.04494242,
#  0.04227121]
    # 1010e4
#     Pd_sim = [0.2891947,  0.17363924 ,0.09941 ,   0.06372652, 0.04950712, 0.04491409,
#  0.04388061]
    # 10101010e3
    # Pd_sim = [0.28957727 ,0.17565303, 0.09822879, 0.06574242, 0.04732424, 0.04832576,
#  0.04291212]
    # # 1010_laplace
    # Pd_sim =  [0.32997273, 0.20857273 ,0.11874394, 0.06840606, 0.04878485 ,0.04548939 ,0.04223485]
    # Pd_theo = [0.28661704, 0.17293213 ,0.10010809 ,0.06297494, 0.04753877, 0.04484079, 0.04228119]
    # energy_estimation = [0.33047273, 0.21051061, 0.11995152, 0.06954848, 0.04968788, 0.0463803, 0.04312879]
    # 1010_gaussian
    Pd_sim =  [0.28560606 ,0.17462727 ,0.09820606 ,0.06198788, 0.0507697,  0.04386061, 0.04158788]
    Pd_theo = [0.28659396, 0.17386366, 0.0975197,  0.06203868, 0.05102629 ,0.04384539 ,0.04158066]
    energy_estimation = [0.28711364, 0.17636364, 0.09960152, 0.06304545 ,0.05178636, 0.04492273 , 0.04228939]
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False
    # print(f"Test Accuracy: {correct / total * 100:.2f}%")
    plt.plot(SNR_dB, Pd_theo, 'o-', color='blue', label='energy_theory')
    plt.plot(SNR_dB, Pd_sim, 'D-', color='purple', label='energy_mc')
    plt.plot(SNR_dB, energy_estimation, '*-', color='green', label='energy_estimation')
    plt.plot(SNR_dB, Pd_modelIQ, '^-', color='cyan', label='model_IQ')
    plt.plot(SNR_dB, Pd_modelSCM, '--', color='red', label='model_SCM')

    plt.yscale('log')

    # 根据您的数据范围设置合适的刻度
    yticks_positions = [  0.05,0.1, 0.4]  # 10⁻³, 10⁻², 10⁻¹, 10⁰
    yticks_labels = [  '5x10$^{-2}$', '10$^{-1}$', '4x10$^{-1}$']

    plt.yticks(yticks_positions, yticks_labels)

    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR (Perfect Sync)')
    plt.grid(True)
    plt.savefig(prefix_s+f'results/CNN'+'_1010.png')
    plt.show()
    # print(Pd_sim)
    # print(Pd_sim_2)
    print(Pd_modelIQ)
    if SCM:
        print(Pd_modelSCM)
    # print(Pd_theo)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with specified configuration"
    )
    # 必要参数
    model_zoo = model_registry.get_all_model_names()
    parser.add_argument('-cuda', '--cuda', type=str, default='1', help='the cuda number')
    # parser.add_argument(
    #     "--model1", type=str, default= 'CNN_150', choices=model_zoo, help="Model to train IQ data"
    # )
    parser.add_argument(
        "--model1", type=str, default= 'CNN_150', choices=model_zoo, help="Model to train IQ data"
    )
    parser.add_argument(
        "--model2", type=str, default= 'CNN_150_S',choices=model_zoo, help="Model to train SCM data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--base_data_dir", type=str, default="./data")
    parser.add_argument(
        "--SNRs", type=int, default=7, help="SNRs number in one model"
    )

    # parser.add_argument(
    #     "--scheduler",
    #     type=str,
    #     required=True,
    #     choices=["reducelronplateau", "onecycle"],
    # )
    # parser.add_argument(
    #     "--use_amp", action="store_true", help="Enable Automatic Mixed Precision (AMP)"
    # )
    # parser.add_argument(
    #     "--use_ddp", action="store_true", help="Enable Distributed Data Parallel (DDP)"
    # )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--val_start_epoch", type=int, default=0, help="Epoch to start validation"
    )
    # parser.add_argument("--seq_len", type=int, required=True, help="Sequence length")
    # parser.add_argument(
    #     "--num_classes", type=int, required=True, help="Number of classes"
    # )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=100,
        help="Stop training if no improvement for this many consecutive epochs",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "moe"],
        help="Loss function to use",
    )

    # 可选参数
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--model_m", type=int, default=None)
    parser.add_argument("--num_periods", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--model_save_dir", type=str, default=None)
    parser.add_argument("--onecycle_div_factor", type=float, default=None)
    parser.add_argument("--onecycle_pct_start", type=float, default=None)
    parser.add_argument("--onecycle_final_div_factor", type=float, default=None)
    parser.add_argument(
        "--pooling", type=str, default=None, choices=["avgpool", "maxpool"]
    )
    parser.add_argument("--amp_init_scale", type=int, default=None)
    parser.add_argument("--merge_mode", type=str, default=None, choices=["OR", "AND"])
    parser.add_argument("--moe_alpha", type=float, default=None)
    args = parser.parse_args()
    main(args)
