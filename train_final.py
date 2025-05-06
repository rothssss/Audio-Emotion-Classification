# Emotion classification pipeline combining extracted features + mel-spectrogram CNN + cross-modal attention (Even Deeper Residual + Dropout2d + Warm Restarts)
# Save this as emotion_combined.py and run directly

import os
import argparse
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

# Data
def prepare_data(dataset_dir, csv_path, test_size=0.025, random_state=42):
    orig_df = pd.read_csv(csv_path)
    df = orig_df[orig_df['Label'] != 'Test'].reset_index(drop=True)
    feature_cols = df.columns.tolist()[2:]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    le = LabelEncoder()
    df['y'] = le.fit_transform(df['Label'])
    df['path'] = df.apply(lambda r: os.path.join(dataset_dir, r['Label'], r['File_name'].replace('.csv','.wav')), axis=1)
    df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)

    features = df[feature_cols].values
    idx = np.arange(len(df))
    tr_idx, val_idx = train_test_split(idx, test_size=test_size, stratify=df['y'], random_state=random_state)
    p_tr = df.loc[tr_idx, 'path'].tolist(); 
    p_val = df.loc[val_idx, 'path'].tolist()
    y_tr = df.loc[tr_idx, 'y'].values;        
    y_val = df.loc[val_idx, 'y'].values
    X_tr = features[tr_idx];                    
    X_val = features[val_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr);         X_val = scaler.transform(X_val)

    return (p_tr, X_tr, y_tr), (p_val, X_val, y_val), le, scaler, len(feature_cols)

# Dataset object
class CombinedDataset(Dataset):
    def __init__(self, paths, feats, labels, sr, duration, n_fft, hop_length, n_mels, max_frames):
        self.paths, self.feats, self.labels = paths, feats, labels
        self.sr, self.duration = sr, duration
        self.n_fft, self.hop_length = n_fft, hop_length
        self.n_mels, self.max_frames = n_mels, max_frames
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        y, _ = librosa.load(self.paths[idx], sr=self.sr, duration=self.duration)
        if len(y) < int(self.duration*self.sr):
            y = np.pad(y, (0,int(self.duration*self.sr)-len(y)))
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < self.max_frames:
            mel_db = np.pad(mel_db, ((0,0),(0,self.max_frames-mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:,:self.max_frames]
        mel_db = (mel_db - mel_db.mean())/(mel_db.std()+1e-6)
        return (torch.from_numpy(mel_db).unsqueeze(0).float(), torch.from_numpy(self.feats[idx]).float(), torch.tensor(self.labels[idx]).long())

# model definition
# NEED TO PASTE INTO TEST PY
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2d = nn.Dropout2d(p=drop_prob)
        self.shortcut = (nn.Sequential(nn.Conv2d(in_ch, out_ch,1,bias=False), nn.BatchNorm2d(out_ch)) if in_ch!=out_ch else nn.Identity())
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout2d(out)
        out += self.shortcut(x)
        return self.relu(out)

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, drop_prob=0.1):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self, x):
        out = self.bn(self.fc(x))
        out = self.dropout(out)
        return self.relu(out + x)

# Combined Model
class CombinedNet(nn.Module):
    def __init__(self, n_mels, feat_dim, num_classes):
        super().__init__()
        # residual conv blocks
        chs = [1,64,64,128,128,256,256,512,512,1024,1024,1024]
        self.convs = nn.ModuleList([ResidualConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2,ceil_mode=True)
        self.gpool = nn.AdaptiveAvgPool2d((1,1))
        # MLP
        self.feat_fc1 = nn.Linear(feat_dim,512)
        self.feat_res = nn.Sequential(ResidualMLPBlock(512), ResidualMLPBlock(512))
        self.feat_dropout = nn.Dropout(0.1)
        self.feat_fc2 = nn.Linear(512,128)
        # Attention
        self.proj_audio = nn.Linear(1024,128)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        # Fusion
        self.fuse_fc1 = nn.Linear(1024+128+128,512)
        self.fuse_res = ResidualMLPBlock(512)
        self.fuse_dropout = nn.Dropout(0.1)
        self.fuse_fc2 = nn.Linear(512, num_classes)
    def forward(self, mel, feat):
        x = mel
        for i, block in enumerate(self.convs):
            x = block(x)
            if i%2==1: x = self.pool(x)
        x_flat = self.gpool(x).view(x.size(0),-1)  
        # feature branch
        f = F.relu(self.feat_fc1(feat)); f = self.feat_res(f); f = self.feat_dropout(f)
        f = F.relu(self.feat_fc2(f))            
        # attention
        f_q = f.unsqueeze(1)
        x_proj = self.proj_audio(x_flat).unsqueeze(1)
        f_att,_ = self.attn(query=f_q,key=x_proj,value=x_proj)
        f_att = f_att.squeeze(1)
        # fusion
        h = torch.cat([x_flat,f,f_att],dim=1)
        h = F.relu(self.fuse_fc1(h)); h = self.fuse_res(h); h = self.fuse_dropout(h)
        return self.fuse_fc2(h)

# train
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_overfit', action='store_true')
    args = parser.parse_args()
    dataset_dir = r"elec378 sp25 dataset/elec378 sp25 dataset"
    csv_path = "C:/Users/roths/Documents/a whats-next/378/final project/ser-mstr/feature_extracted_combinedfinal.csv"
    sr, duration = 16000,3.0; n_fft, hop_length, n_mels = 1024,512,64
    max_frames = int(np.ceil(duration*sr/hop_length))
    batch_size = 16 if args.debug_overfit else 32
    num_epochs=80
    patience = 25

    (p_tr,X_tr,y_tr),(p_val,X_val,y_val),le,scaler,feat_dim = prepare_data(dataset_dir,csv_path)
    num_classes = len(le.classes_)

    ds_kwargs = dict(sr=sr, duration=duration, n_fft=n_fft,
                     hop_length=hop_length, n_mels=n_mels, max_frames=max_frames)
    train_ds = CombinedDataset(p_tr,X_tr,y_tr,**ds_kwargs)
    val_ds   = CombinedDataset(p_val,X_val,y_val,**ds_kwargs)
    #if args.debug_overfit:
    #    train_ds = Subset(train_ds,range(min(16,len(train_ds))))
    #    val_ds   = Subset(val_ds,  range(min(16,len(val_ds))))
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size,shuffle=False,num_workers=0)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CombinedNet(n_mels,feat_dim,num_classes).to(device)

    weights = compute_class_weight('balanced',classes=np.unique(y_tr),y=y_tr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),lr=2e-4,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
    best_val_acc = 0.0
    best_val,float_inf = float('inf'), float('inf'); counter=0
    for epoch in range(1,num_epochs+1):
        model.train(); tloss=0; tacc_count=0; tcount=0
        for mel,feat,yb in tqdm(train_loader,desc=f"Epoch {epoch}▶Train"):
            mel,feat,yb = mel.to(device),feat.to(device),yb.to(device)
            optimizer.zero_grad(); out=model(mel,feat); loss=criterion(out,yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); scheduler.step(epoch + mel.size(0)/len(train_loader))
            tloss+=loss.item()*yb.size(0); preds=out.argmax(1); tacc_count+=(preds==yb).sum().item(); tcount+=yb.size(0)
        train_acc=tacc_count/tcount; train_loss=tloss/tcount
        model.eval(); vloss=0; vacc_count=0; vcount=0
        with torch.no_grad():
            for mel,feat,yb in tqdm(val_loader,desc=f"Epoch {epoch}▶Val"):
                mel,feat,yb=mel.to(device),feat.to(device),yb.to(device)
                out=model(mel,feat); loss=criterion(out,yb)
                vloss+=loss.item()*yb.size(0); preds=out.argmax(1); vacc_count+=(preds==yb).sum().item(); vcount+=yb.size(0)
        val_acc=vacc_count/vcount; val_loss=vloss/vcount
        print(f"Epoch{epoch}: tr_loss={train_loss:.3f}, tr_acc={train_acc:.3f}, val_loss={val_loss:.3f}, val_acc={val_acc:.3f}")
        if val_acc> best_val_acc:
            #best_val=val_loss; counter=0; torch.save(model.state_dict(),'best_extreme.pt')
            best_val_acc=val_acc
            counter = 0
            torch.save(model.state_dict(),'best_extreme.pt')      
            print("Saved best model")      
        else:
            counter+=1
            if counter>=patience:
                print("Early stopping")
                break
    print("Training complete")

if __name__=='__main__':
    from multiprocessing import freeze_support; freeze_support(); main()
