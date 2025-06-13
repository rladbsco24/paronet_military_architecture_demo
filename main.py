import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 1. Operator Classes ----------
class FFTOperator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out_dim, in_dim, dtype=torch.cfloat) * 0.05)
    def forward(self, x):
        x_c = torch.complex(x, torch.zeros_like(x)) if not torch.is_complex(x) else x
        x_fft = torch.fft.fft(x_c, dim=1)
        x_fft_out = x_fft @ self.kernel.t()
        x_out = torch.fft.ifft(x_fft_out, dim=1)
        return torch.real(x_out)

class FNOOperator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 48), nn.GELU(),
            nn.Linear(48, 48), nn.GELU(),
            nn.Linear(48, out_dim)
        )
    def forward(self, x):
        return self.head(x)

class PSDTransformerNOOperator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.trunk(x)

class RegionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# --------- 2. 확장 Region Mask 생성 함수 ----------
def create_expanded_region_mask(coords, situation_flags, sensor_status, threat_level, region_defs):
    """
    coords: (N, 3)
    situation_flags: (N,) int flag [0:normal, 1:위험, 2:교란, 3:센서장애]
    sensor_status: (N,) float (0~1, 신뢰도)
    threat_level: (N,) float (0~1, 위협도)
    region_defs: dict (geometry zone의 center/radius 등)
    Return: (N,) region_mask
    """
    N = coords.shape[0]
    region_mask = torch.zeros(N, dtype=torch.long, device=coords.device)
    # 1. geometry region (target zone) : 좌표 중심/반경
    dists = torch.norm(coords - torch.tensor(region_defs['target_center'], dtype=coords.dtype, device=coords.device), dim=1)
    mask_geometry = dists < region_defs['target_radius']
    region_mask[mask_geometry] = 1

    # 2. 상황 기반 위험 region (위협도+상황 flag)
    mask_threat = (threat_level > 0.7) | (situation_flags == 1)
    region_mask[mask_threat & ~mask_geometry] = 2

    # 3. 교란 zone (교란 flag)
    mask_jammer = (situation_flags == 2)
    region_mask[mask_jammer & ~(mask_geometry | mask_threat)] = 3

    # 4. 센서 장애 zone (신뢰도↓/flag)
    mask_sensor = (sensor_status < 0.4) | (situation_flags == 3)
    region_mask[mask_sensor & ~(mask_geometry | mask_threat | mask_jammer)] = 4

    # 0: 나머지는 background (geometry/situation 모두 해당X)
    return region_mask

# --------- 3. SoftRegionOperator ----------
class SoftRegionOperator(nn.Module):
    def __init__(self, in_dim, out_dim, n_regions=5):
        super().__init__()
        self.ops = nn.ModuleList([
            FFTOperator(in_dim, out_dim),               # 0: background
            FNOOperator(in_dim, out_dim),               # 1: geometry target region
            PSDTransformerNOOperator(in_dim, out_dim),  # 2: 위험(상황기반)
            RegionMLP(in_dim, out_dim),                 # 3: 교란
            RegionMLP(in_dim, out_dim)                  # 4: 센서장애 zone
        ])
        self.n_regions = n_regions

    def forward(self, x, region_mask):
        outs = torch.zeros(x.shape[0], self.ops[0].kernel.shape[0], device=x.device)
        for i in range(self.n_regions):
            idxs = (region_mask == i)
            if torch.sum(idxs) > 0:
                outs[idxs] = self.ops[i](x[idxs])
        return outs

# --------- 4. Main PARONet Model ----------
class PARONetExpRegion(nn.Module):
    def __init__(self, in_dim=8, out_dim=2, n_regions=5):
        super().__init__()
        self.soft_op = SoftRegionOperator(in_dim, out_dim, n_regions)
        self.coord_encoder = nn.Linear(3, in_dim-5)  # (coords + 상황/센서 feature)

    def forward(self, coords, situation_feat, region_mask):
        # situation_feat: (N, 5) (threat_level, sensor_status, 3개 flag onehot)
        feat = torch.cat([self.coord_encoder(coords), situation_feat], dim=1)
        out = self.soft_op(feat, region_mask)
        return out

# --------- 5. 데이터/환경 생성 ----------
def create_sim_data(N=2400, device='cpu'):
    # 좌표 분포 (x/y/z)
    x = np.random.uniform(-2, 2, N)
    y = np.random.uniform(-2, 2, N)
    z = np.random.uniform(0, 10, N)
    coords = torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32, device=device)

    # 상황/센서 정보
    threat_level = torch.clamp(torch.randn(N, device=device)*0.3 + 0.5, 0, 1)
    sensor_status = torch.clamp(torch.randn(N, device=device)*0.2 + 0.8, 0, 1)
    # 상황 flag: [0=normal, 1=위협, 2=교란, 3=센서장애]
    situation_flags = torch.randint(0, 4, (N,), device=device)
    # flag onehot
    situation_onehot = F.one_hot(situation_flags, num_classes=4).float()

    # 상황 feature: (threat_level, sensor_status, 상황flag onehot)
    situation_feat = torch.cat([threat_level.unsqueeze(1), sensor_status.unsqueeze(1), situation_onehot], dim=1)

    # region 정의 (target zone geometry)
    region_defs = {
        'target_center': [0.5, 0.5, 8.5],
        'target_radius': 0.8,
    }
    return coords, situation_feat, situation_flags, sensor_status, threat_level, region_defs

# --------- 6. Ground Truth 생성 (성능 평가용 문제) ----------
def create_ground_truth(coords, region_mask):
    gt = torch.zeros(coords.shape[0], 2, device=coords.device)
    # 0: field, 1: target id
    # geometry region: 정확한 target field + id
    gt[region_mask==1,0] = torch.exp(-((coords[region_mask==1,0]-0.5)**2 + (coords[region_mask==1,1]-0.5)**2 + (coords[region_mask==1,2]-8.5)**2)/0.12) + 0.08*torch.randn_like(coords[region_mask==1,0])
    gt[region_mask==1,1] = 1.0

    # 위험 region(상황 기반): 고주파+잡음 field, id는 0
    gt[region_mask==2,0] = 0.6*torch.sin(12*coords[region_mask==2,2]) + 0.3*torch.randn_like(coords[region_mask==2,0])
    gt[region_mask==2,1] = 0.0

    # 교란 region: 불규칙 진폭 field, id 0
    gt[region_mask==3,0] = 0.5*torch.cos(6*coords[region_mask==3,2]) + 0.2*torch.randn_like(coords[region_mask==3,0])
    gt[region_mask==3,1] = 0.0

    # 센서 장애 region: random field, id 0
    gt[region_mask==4,0] = 0.3*torch.rand(coords[region_mask==4,0].shape[0], device=coords.device)
    gt[region_mask==4,1] = 0.0

    # background: 약한 noise field, id 0
    gt[region_mask==0,0] = 0.08 * torch.sin(coords[region_mask==0,2]) + 0.05*torch.randn_like(coords[region_mask==0,0])
    gt[region_mask==0,1] = 0.0
    return gt

# --------- 7. Loss 함수 ----------
def multi_loss(pred, gt, region_mask):
    loss_target = F.mse_loss(pred[region_mask==1,0], gt[region_mask==1,0]) + F.binary_cross_entropy_with_logits(pred[region_mask==1,1], gt[region_mask==1,1])
    loss_risk = F.mse_loss(pred[region_mask==2,1], gt[region_mask==2,1])
    loss_jammer = F.mse_loss(pred[region_mask==3,1], gt[region_mask==3,1])
    loss_sensor = F.mse_loss(pred[region_mask==4,1], gt[region_mask==4,1])
    loss_bg = F.mse_loss(pred[region_mask==0,1], gt[region_mask==0,1])
    total_loss = loss_target + 0.08*(loss_risk + loss_jammer + loss_sensor + loss_bg)
    return total_loss

# --------- 8. 학습/분석/시각화 ---------
def train_paronet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coords, situation_feat, situation_flags, sensor_status, threat_level, region_defs = create_sim_data(2400, device=device)
    region_mask = create_expanded_region_mask(coords, situation_flags, sensor_status, threat_level, region_defs)
    gt = create_ground_truth(coords, region_mask)
    model = PARONetExpRegion(in_dim=8, out_dim=2, n_regions=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    n_epochs = 300
    for epoch in range(n_epochs):
        model.train()
        pred = model(coords, situation_feat, region_mask)
        loss = multi_loss(pred, gt, region_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Total Loss: {loss.item():.5f}")
    print("학습 종료")

    # 성능 평가/시각화
    with torch.no_grad():
        pred = model(coords, situation_feat, region_mask)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,5))
        plt.scatter(coords.cpu().numpy()[:,2], pred[:,0].cpu().numpy(), c=region_mask.cpu().numpy(), cmap='jet', alpha=0.6, s=10, label="Pred Field")
        plt.scatter(coords.cpu().numpy()[:,2], gt[:,0].cpu().numpy(), c='k', alpha=0.2, s=8, label="GT Field")
        plt.title("Predicted Field Strength vs GT")
        plt.xlabel("z (penetration axis)")
        plt.ylabel("field")
        plt.show()

        # 타겟/비타겟 분리
        plt.figure(figsize=(6,4))
        plt.hist(torch.sigmoid(pred[region_mask==1,1]).cpu().numpy(), bins=25, alpha=0.7, label='Target (pred)')
        plt.hist(torch.sigmoid(pred[region_mask!=1,1]).cpu().numpy(), bins=25, alpha=0.7, label='Non-target (pred)')
        plt.title("Target/Non-target Class Separation (Sigmoid)")
        plt.xlabel("Predicted Target ID")
        plt.legend()
        plt.show()

        # region별 예측 MSE (성능 지표)
        print("\n[Region별 field 예측 MSE]")
        for i in range(5):
            mse = F.mse_loss(pred[region_mask==i,0], gt[region_mask==i,0]).item() if torch.sum(region_mask==i)>0 else None
            print(f"region {i}: MSE={mse}")

        # Precision/Recall in Target region
        pred_label = (torch.sigmoid(pred[:,1]) > 0.5).long()
        true_label = (gt[:,1] > 0.5).long()
        tp = torch.sum((pred_label==1) & (true_label==1)).item()
        fp = torch.sum((pred_label==1) & (true_label==0)).item()
        fn = torch.sum((pred_label==0) & (true_label==1)).item()
        prec = tp / (tp+fp+1e-6)
        recall = tp / (tp+fn+1e-6)
        print(f"\nTarget region: Precision={prec:.3f}, Recall={recall:.3f}, TP={tp}, FP={fp}, FN={fn}")

    return model, coords, situation_feat, region_mask, gt

if __name__ == "__main__":
    train_paronet()
