import argparse, os, glob, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------- utils --------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_ts_col(df, ts_col):
    if np.issubdtype(df[ts_col].dtype, np.number):
        return df[ts_col].astype('int64').values
    ts = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    return (ts.view('int64') // 10**6).astype('int64')

def detect_feature_cols(df, ts_col):
    cols = [c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric feature columns detected. Specify --features.")
    return cols

def zfit(X): mu, sd = np.nanmean(X,0), np.nanstd(X,0)+1e-8; return mu, sd
def zapply(X, mu, sd): return ((X-mu)/sd).astype('float32')

# -------- light augmentations --------

class TimeAug:
    def __init__(self, jitter=0.01, scale=0.1, tmask=0.05):
        self.jitter, self.scale, self.tmask = jitter, scale, tmask
    def __call__(self, x):
        # x: [B,C,T]
        B,C,T = x.shape; y=x
        if self.scale>0:
            y = y * (torch.randn(B,C,1, device=x.device)*self.scale + 1.0)
        if self.jitter>0:
            y = y + torch.randn_like(y)*self.jitter
        L = int(T*self.tmask)
        if L>0:
            s = torch.randint(0, T-L+1, (B,), device=x.device)
            for b in range(B): y[b,:,s[b]:s[b]+L]=0
        return y

# -------- encoder (TCN-like) + projection --------

class ResBlock1D(nn.Module):
    def __init__(self, c, d, k=3, p=0.1):
        super().__init__()
        pad=(k-1)*d//2
        self.c1=nn.Conv1d(c,c,k,padding=pad,dilation=d); self.b1=nn.BatchNorm1d(c)
        self.c2=nn.Conv1d(c,c,k,padding=pad,dilation=d); self.b2=nn.BatchNorm1d(c)
        self.drop=nn.Dropout(p)
    def forward(self,x):
        h=F.gelu(self.b1(self.c1(x))); h=self.drop(h); h=self.b2(self.c2(h))
        return F.gelu(x+h)

class TSEncoder(nn.Module):
    def __init__(self, in_ch, hidden=256, depth=4, emb_dim=128, proj_dim=128, drop=0.1):
        super().__init__()
        self.inp=nn.Conv1d(in_ch,hidden,3,padding=1)
        self.blocks=nn.Sequential(*[ResBlock1D(hidden,2**i,3,drop) for i in range(depth)])
        self.head=nn.Conv1d(hidden,emb_dim,1)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.proj=nn.Sequential(nn.Linear(emb_dim,proj_dim), nn.GELU(), nn.Linear(proj_dim,proj_dim))
    def forward(self,x):
        h=F.gelu(self.inp(x)); h=self.blocks(h); h=self.head(h)   # [B, emb_dim, T]
        g=self.pool(h).squeeze(-1)                                # [B, emb_dim]
        z=F.normalize(self.proj(g), dim=-1)                       # [B, proj_dim]
        return g,z

# -------- NT-Xent --------

def ntxent(z1,z2,tau=0.2):
    B=z1.size(0); z=torch.cat([z1,z2],0); sim=(z@z.t())/tau
    sim.fill_diagonal_(-9e15)
    pos=torch.cat([torch.arange(B,2*B),torch.arange(0,B)]).to(z.device)
    return F.cross_entropy(sim, pos)

# -------- dataset (sliding windows + two views) --------

class WinDS(Dataset):
    def __init__(self,X,ts,W,S,aug):
        self.X = X
        self.ts = np.asarray(ts)
        self.W,self.S,self.aug=W,S,aug
        self.ends=np.arange(W,len(X),S,dtype=np.int64)
    def __len__(self): return len(self.ends)
    def __getitem__(self,i):
        e=int(self.ends[i]); s=e-self.W; x=self.X[s:e].T
        x=torch.from_numpy(x.astype('float32'))
        x1=self.aug(x.unsqueeze(0)).squeeze(0); x2=self.aug(x.unsqueeze(0)).squeeze(0)
        return x1,x2,int(self.ts[e-1])

# -------- train + export --------

def train_symbol(args, sym):
    # 1) Load CSV shards
    in_dir = os.path.join(args.data_root, sym)
    files = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV at {args.data_root}/{sym}/ (expected YYYY-MM-DD.csv)")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # 2) Timestamps & features
    ts = parse_ts_col(df, args.ts_col)
    feats = detect_feature_cols(df,args.ts_col) if args.features=="auto" else [c.strip() for c in args.features.split(",")]
    for c in feats:
        if c not in df.columns:
            raise KeyError(f"Missing feature: {c}")
    X = df[feats].astype('float32').values

    # 3) Optional 1s resample
    if args.resample_1s:
        t=pd.to_datetime(ts,unit='ms',utc=True)
        tmp=pd.DataFrame(X,index=t,columns=feats).resample('1S').ffill()
        ts=(tmp.index.view('int64')//10**6).astype('int64'); X=tmp.values.astype('float32')

    # 4) Split, z-score by train
    N=len(X); assert N>args.window+args.batch_size, f"Not enough rows {N}"
    split=int(N*0.9); Xtr,tstr=X[:split],ts[:split]; Xva,tsva=X[split:],ts[split:]
    mu,sd=zfit(Xtr); Xtr=zapply(Xtr,mu,sd); Xva=zapply(Xva,mu,sd)

    # 5) Datasets
    aug=TimeAug(args.aug_jitter,args.aug_scale,args.aug_time_mask)
    dtr=WinDS(Xtr,tstr,args.window,args.train_stride,aug)
    dva=WinDS(Xva,tsva,args.window,args.train_stride,aug)
    # Small validation slices (len < window) can yield zero batches; fall back to
    # training split for validation in that case so the loop stays well-defined.
    if len(dva) == 0:
        Xva, tsva = Xtr, tstr
        dva = WinDS(Xva, tsva, args.window, args.train_stride, aug)
    dltr=DataLoader(dtr,batch_size=args.batch_size,shuffle=True,drop_last=True)
    dlva=DataLoader(dva,batch_size=args.batch_size,shuffle=False,drop_last=True)

    # 6) Model
    device=torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device)
    enc=TSEncoder(len(feats),args.hidden_dim,args.depth,args.emb_dim,args.proj_dim,args.dropout).to(device)
    opt=torch.optim.AdamW(enc.parameters(),lr=args.lr,weight_decay=1e-4)

    # 7) Train
    best=float('inf')
    for ep in range(1,args.epochs+1):
        enc.train(); tr=0.0
        for x1,x2,_ in dltr:
            x1,x2=x1.to(device),x2.to(device); opt.zero_grad(set_to_none=True)
            _,z1=enc(x1); _,z2=enc(x2); loss=ntxent(z1,z2,args.tau)
            loss.backward(); opt.step(); tr+=loss.item()*x1.size(0)
        tr/=len(dtr)
        enc.eval(); va=0.0
        with torch.no_grad():
            for x1,x2,_ in dlva:
                x1,x2=x1.to(device),x2.to(device)
                _,z1=enc(x1); _,z2=enc(x2); va+=ntxent(z1,z2,args.tau).item()*x1.size(0)
        va/=len(dva)
        print(f"[{sym}] epoch {ep}/{args.epochs} train={tr:.4f} valid={va:.4f}")
        if va<best:
            best=va; ensure_dir(args.model_out)
            torch.save(enc.state_dict(), os.path.join(args.model_out, f"ts2vec_{sym}.pt"))
            meta={"version":"0.1.0","symbol":sym,"features_order":feats,
                  "normalization":{"mean":mu.tolist(),"std":sd.tolist()},
                  "window":args.window,"emb_dim":args.emb_dim,"proj_dim":args.proj_dim,
                  "max_lag_ms":args.max_lag_ms}
            with open(os.path.join(args.model_out, f"ts2vec_{sym}.meta.json"),"w",encoding="utf-8") as f:
                json.dump(meta,f,ensure_ascii=False,indent=2)

    # 8) Export embeddings over full series (reload best ckpt)
    enc.eval(); ck=os.path.join(args.model_out,f"ts2vec_{sym}.pt")
    if os.path.exists(ck): enc.load_state_dict(torch.load(ck,map_location=device))

    X_all = zapply(df[feats].astype('float32').values, mu, sd)
    ts_all = parse_ts_col(df, args.ts_col)
    W=args.window; step=args.export_stride; ends=range(W,len(X_all),step)

    outs, tss = [], []
    with torch.no_grad():
        B=args.batch_size; buf_x, buf_t = [], []
        for e in ends:
            s=e-W; buf_x.append(X_all[s:e].T.astype('float32')); buf_t.append(int(ts_all[e-1]))
            if len(buf_x)==B:
                xt=torch.from_numpy(np.stack(buf_x)).to(device)
                g,_=enc(xt); outs.append(g.cpu().numpy().astype('float32')); tss.extend(buf_t)
                buf_x,buf_t = [], []
        if buf_x:
            xt=torch.from_numpy(np.stack(buf_x)).to(device)
            g,_=enc(xt); outs.append(g.cpu().numpy().astype('float32')); tss.extend(buf_t)

    emb = np.concatenate(outs,0) if outs else np.zeros((0,args.emb_dim),dtype='float32')
    tsA = np.array(tss,dtype='int64')

    out_dir=os.path.join(args.out_root, sym); ensure_dir(out_dir)
    # daily shards + manifest
    if len(tsA)>0:
        dt=pd.to_datetime(tsA,unit='ms',utc=True).strftime('%Y-%m-%d')
        dfi=pd.DataFrame({'ts':tsA,'date':dt})
        for date, idxs in dfi.groupby('date').groups.items():
            idx=np.array(sorted(list(idxs)))
            np.savez_compressed(os.path.join(out_dir,f'{date}.npz'), ts=tsA[idx], emb=emb[idx])
        mani={"symbol":sym,"embedding_dim":args.emb_dim,"shards":[]}
        for fn in sorted(glob.glob(os.path.join(out_dir,"*.npz"))):
            arr=np.load(fn); rows=int(arr['ts'].shape[0])
            mani['shards'].append({"date":os.path.basename(fn)[:-4],"rows":rows,
                                   "ts_min":int(arr['ts'].min()) if rows else None,
                                   "ts_max":int(arr['ts'].max()) if rows else None})
        with open(os.path.join(out_dir,"manifest.json"),"w",encoding="utf-8") as f:
            json.dump(mani,f,ensure_ascii=False,indent=2)
    print(f"[{sym}] embeddings -> {out_dir} ; model -> {args.model_out}")

def main():
    ap=argparse.ArgumentParser("TS2Vec minimal trainer")
    ap.add_argument("--symbols", required=True, help="CSV of symbols, e.g. BTCUSDT,SOLUSDT,HYPEUSDT")
    ap.add_argument("--data-root", required=True, help="Input root {SYMBOL}/YYYY-MM-DD.csv")
    ap.add_argument("--out-root", default="data/embeddings/ts2vec")
    ap.add_argument("--model-out", default="models/ts2vec")
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--features", default="auto")
    ap.add_argument("--resample-1s", action="store_true")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--train-stride", type=int, default=8)
    ap.add_argument("--export-stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aug-jitter", type=float, default=0.01)
    ap.add_argument("--aug-scale", type=float, default=0.05)
    ap.add_argument("--aug-time-mask", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")   # 'auto'|'cuda'|'cpu'
    ap.add_argument("--max-lag-ms", type=int, default=200)
    args=ap.parse_args()
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        train_symbol(args, sym)

if __name__=="__main__":
    main()
