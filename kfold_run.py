import sys, io, json, re, contextlib, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, linregress
from src.data_loader import load_raw_real_data, create_scaled_dataloaders
from src.model import DualStreamEntropyClockNet
from src.train import train_model

protocol = sys.argv[1]            # 'A' = repo CSVs (filter+impute before split), 'B' = filter+impute inside each fold
seed = int(sys.argv[2])
K, EPOCHS, REPS = 5, 100, 100
torch.set_num_threads(4)

def raw_matrices():
    def clean_id(c): return re.sub(r'_S\d+_L\d+$', '', c.replace('.bam', ''))
    m = pd.read_csv('data/raw/GSE288139_buccalswab_avgmeth.txt.gz', index_col=0)
    e = pd.read_csv('data/raw/GSE288139_buccalswab_entropy.txt.gz', index_col=0)
    m.columns = m.columns.map(clean_id); e.columns = e.columns.map(clean_id)
    m, e = m.T, e.T
    ages = pd.read_csv('data/processed/clean_ages.csv', index_col=0)
    common = m.index.intersection(e.index).intersection(ages.index)
    return m.loc[common], e.loc[common], ages.loc[common]

if protocol == 'A':
    meth_df, ent_df, ages_df = load_raw_real_data()
else:
    meth_df, ent_df, ages_df = raw_matrices()
y = ages_df['Age'].values
idx = np.arange(len(y))
bins = np.digitize(y, bins=np.percentile(y, [25, 50, 75]))
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

oof = {k: np.full(len(y), np.nan) for k in ('dual', 'enet', 'w_meth', 'w_ent')}
folds = []
for f, (tr, te) in enumerate(skf.split(idx, bins)):
    t0 = time.time()
    torch.manual_seed(seed * 100 + f); np.random.seed(seed * 100 + f)
    if protocol == 'B':
        # filter and imputer use the training fold only
        thr = len(tr) * 0.8
        keep_m = meth_df.iloc[tr].dropna(axis=1, thresh=thr).columns
        keep_e = ent_df.iloc[tr].dropna(axis=1, thresh=thr).columns
        loci = keep_m.intersection(keep_e)
        mi, ei = KNNImputer(n_neighbors=5), KNNImputer(n_neighbors=5)
        M = pd.DataFrame(index=meth_df.index, columns=loci, dtype=float)
        E = pd.DataFrame(index=ent_df.index, columns=loci, dtype=float)
        M.iloc[tr] = mi.fit_transform(meth_df.iloc[tr][loci]); M.iloc[te] = mi.transform(meth_df.iloc[te][loci])
        E.iloc[tr] = ei.fit_transform(ent_df.iloc[tr][loci]); E.iloc[te] = ei.transform(ent_df.iloc[te][loci])
        Mf, Ef = M, E
    else:
        Mf, Ef = meth_df, ent_df
    d = Mf.shape[1]
    itr, iva = train_test_split(tr, test_size=0.2, random_state=42, stratify=bins[tr])
    trl, val, tel, ascaler = create_scaled_dataloaders(Mf, Ef, ages_df, itr, iva, te)
    model = DualStreamEntropyClockNet(input_dim=d)
    with contextlib.redirect_stdout(io.StringIO()):
        model = train_model(model, trl, val, epochs=EPOCHS, patience=15)
    model.eval()
    mt, et, at = [], [], []
    for a, b, c in tel: mt.append(a); et.append(b); at.append(c)
    mt, et = torch.cat(mt), torch.cat(et)
    def predict(m_, e_):
        with torch.no_grad():
            return ascaler.inverse_transform(model(m_, e_).numpy().reshape(-1, 1)).flatten()
    p = predict(mt, et)
    with torch.no_grad():
        mo, eo = model.meth_stream(mt), model.ent_stream(et)
        w = model.attention(torch.cat([mo, eo], 1)).numpy()
    yt = y[te]
    base = mean_absolute_error(yt, p)
    rng = np.random.default_rng(seed * 100 + f)
    inc_e, inc_m = [], []
    for _ in range(REPS):
        inc_e.append(mean_absolute_error(yt, predict(mt, et[rng.permutation(len(te))])) - base)
        inc_m.append(mean_absolute_error(yt, predict(mt[rng.permutation(len(te))], et)) - base)
    # ElasticNetCV on the identical fold
    Xa = pd.concat([Mf.iloc[tr], Ef.iloc[tr]], axis=1).values; Xb = pd.concat([Mf.iloc[te], Ef.iloc[te]], axis=1).values
    sc = StandardScaler().fit(Xa)
    en = ElasticNetCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1).fit(sc.transform(Xa), y[tr])
    pe = en.predict(sc.transform(Xb))
    oof['dual'][te] = p; oof['enet'][te] = pe; oof['w_meth'][te] = w[:, 0]; oof['w_ent'][te] = w[:, 1]
    folds.append(dict(fold=f + 1, n_test=len(te), n_loci=d, dual_mae=base, enet_mae=mean_absolute_error(yt, pe),
                      w_ent=float(w[:, 1].mean()), perm_ent=float(np.mean(inc_e)), perm_meth=float(np.mean(inc_m)), secs=round(time.time() - t0)))
    print(protocol, seed, folds[-1], flush=True)

def summ(p):
    r = pearsonr(y, p)[0]; s = linregress(y, p)
    return dict(mae=mean_absolute_error(y, p), r=r, r2=r2_score(y, p), slope=s.slope, intercept=s.intercept)
res = dict(protocol=protocol, seed=seed, folds=folds, dual=summ(oof['dual']), enet=summ(oof['enet']),
           mean_w_ent=float(np.mean(oof['w_ent'])), n_ent_majority=int((oof['w_ent'] > 0.5).sum()),
           w_ent_min=float(oof['w_ent'].min()), w_ent_max=float(oof['w_ent'].max()), w_ent_sd=float(oof['w_ent'].std()),
           perm_ent=float(np.mean([x['perm_ent'] for x in folds])), perm_meth=float(np.mean([x['perm_meth'] for x in folds])),
           perm_ent_by_fold=[x['perm_ent'] for x in folds], perm_meth_by_fold=[x['perm_meth'] for x in folds],
           corr_w_age=float(pearsonr(y, oof['w_ent'])[0]))
json.dump(res, open(f'res_{protocol}_{seed}.json', 'w'), indent=1)
print('DONE', protocol, seed, json.dumps({k: res[k] for k in ('dual', 'enet', 'mean_w_ent', 'n_ent_majority', 'perm_ent', 'perm_meth')}), flush=True)
