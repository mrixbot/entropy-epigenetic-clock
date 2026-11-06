# Cross-validation of the dual-stream model over several training seeds (written for the thesis, run from the repo root).
# Stratified 5-fold CV of the dual-stream network, using the repo's own model/train/data_loader code.
#  variant "repo"   : exactly what run_kfold.py does (reads data/processed, which was filtered + imputed on all 100)
#  variant "strict" : missingness filter + KNN imputation redone inside each fold, from the raw GEO matrices
import sys, io, re, json, contextlib, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.stats import pearsonr
sys.path.insert(0, '.')
from src.data_loader import load_raw_real_data, create_scaled_dataloaders
from src.model import DualStreamEntropyClockNet
from src.train import train_model

def load_raw():
    def clean_id(c): return re.sub(r'_S\d+_L\d+$', '', c.replace('.bam', ''))
    m = pd.read_csv('data/raw/GSE288139_buccalswab_avgmeth.txt.gz', index_col=0)
    e = pd.read_csv('data/raw/GSE288139_buccalswab_entropy.txt.gz', index_col=0)
    m.columns = m.columns.map(clean_id); e.columns = e.columns.map(clean_id)
    m, e = m.T, e.T
    ages = pd.read_csv('data/processed/clean_ages.csv', index_col=0)
    common = m.index.intersection(e.index).intersection(ages.index)
    return m.loc[common], e.loc[common], ages.loc[common]

def strict_prep(m_raw, e_raw, tr, te, thr=0.2):
    need = len(tr) * (1 - thr)
    m_tr, e_tr = m_raw.iloc[tr], e_raw.iloc[tr]
    keep = m_tr.dropna(axis=1, thresh=need).columns.intersection(e_tr.dropna(axis=1, thresh=need).columns)
    out = []
    for raw in (m_raw, e_raw):
        imp = KNNImputer(n_neighbors=5)
        imp.fit(raw.iloc[tr][keep])
        full = pd.DataFrame(index=raw.index, columns=keep, dtype=float)
        full.iloc[tr] = imp.transform(raw.iloc[tr][keep])
        full.iloc[te] = imp.transform(raw.iloc[te][keep])
        out.append(full)
    return out[0], out[1]

def attn_and_preds(model, loader):
    model.eval(); P, T, W = [], [], []
    with torch.no_grad():
        for m, e, a in loader:
            P += model(m, e).numpy().flatten().tolist(); T += a.numpy().flatten().tolist()
            att = model.attention(torch.cat([model.meth_stream(m), model.ent_stream(e)], 1))
            W += att[:, 1].numpy().tolist()
    return np.array(P), np.array(T), np.array(W)

def perm_cost(model, loader, scaler, base_mae, reps=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    Ms, Es, A = [], [], []
    for m, e, a in loader: Ms.append(m); Es.append(e); A.append(a)
    M, E, A = torch.cat(Ms), torch.cat(Es), torch.cat(A).numpy().flatten()
    y = scaler.inverse_transform(A.reshape(-1, 1)).flatten()
    out = {}
    for name in ('ent', 'meth'):
        c = []
        for _ in range(reps):
            idx = torch.randperm(len(M), generator=g)
            with torch.no_grad():
                p = model(M, E[idx]) if name == 'ent' else model(M[idx], E)
            c.append(mean_absolute_error(y, scaler.inverse_transform(p.numpy().reshape(-1, 1)).flatten()) - base_mae)
        out[name] = float(np.mean(c))
    return out

def run(variant, seed, k=5, epochs=50):
    torch.manual_seed(seed); np.random.seed(seed)
    if variant == 'repo':
        m_df, e_df, ages = load_raw_real_data()
    else:
        m_raw, e_raw, ages = load_raw()
    y = ages['Age'].values; n = len(y)
    bins = np.digitize(y, bins=np.percentile(y, [25, 50, 75]))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    R = dict(mae=[], r=[], r2=[], w_ent=[], pm_ent=[], pm_meth=[], en_mae=[], en_r=[], loci=[], ind_w=np.zeros(n).tolist())
    for fold, (tr, te) in enumerate(skf.split(np.arange(n), bins)):
        itr, iva = train_test_split(tr, test_size=0.2, random_state=42, stratify=bins[tr])
        if variant == 'strict':
            m_df, e_df = strict_prep(m_raw, e_raw, tr, te)
        R['loci'].append(m_df.shape[1])
        trl, val, tel, sc = create_scaled_dataloaders(m_df, e_df, ages, itr, iva, te)
        model = DualStreamEntropyClockNet(input_dim=m_df.shape[1]); model.age_scaler = sc
        with contextlib.redirect_stdout(io.StringIO()):
            model = train_model(model, trl, val, epochs=epochs, patience=15)
        P, T, W = attn_and_preds(model, tel)
        P = sc.inverse_transform(P.reshape(-1, 1)).flatten(); T = sc.inverse_transform(T.reshape(-1, 1)).flatten()
        mae = mean_absolute_error(T, P)
        R['mae'].append(mae); R['r'].append(pearsonr(T, P)[0]); R['r2'].append(r2_score(T, P))
        R['w_ent'].append(float(W.mean()))
        for j, i in enumerate(te): R['ind_w'][i] = float(W[j])
        pc = perm_cost(model, tel, sc, mae, seed=seed); R['pm_ent'].append(pc['ent']); R['pm_meth'].append(pc['meth'])
        # ElasticNetCV baseline on the same outer fold (train = inner-train + inner-val)
        X = pd.concat([m_df, e_df], axis=1).values
        ss = StandardScaler().fit(X[tr]); en = ElasticNetCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1).fit(ss.transform(X[tr]), y[tr])
        pe = en.predict(ss.transform(X[te]))
        R['en_mae'].append(mean_absolute_error(y[te], pe)); R['en_r'].append(pearsonr(y[te], pe)[0])
    R['frac_ent_gt50'] = float(np.mean(np.array(R['ind_w']) > 0.5))
    return R

if __name__ == '__main__':
    variant = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(',')]
    res = {}
    for s in seeds:
        t = time.time(); res[s] = run(variant, s)
        r = res[s]; print(f"{variant} seed {s}: MAE {np.mean(r['mae']):.2f}  r {np.mean(r['r']):.3f}  ent-w {np.mean(r['w_ent'])*100:.1f}%  EN MAE {np.mean(r['en_mae']):.2f}  ({time.time()-t:.0f}s)", flush=True)
        json.dump(res, open(f'res_{variant}.json', 'w'))
