import numpy as np
import pandas as pd
from pyreghdfe import reghdfe, ivreghdfe


def _dummy_ols(df, y, xs, fes):
    D = pd.get_dummies(df[fes].astype("category"), drop_first=False, dtype=float)
    A = np.column_stack([df[xs].to_numpy(float), D.to_numpy(float)])
    b, *_ = np.linalg.lstsq(A, df[y].to_numpy(float), rcond=None)
    return b[:len(xs)]


def test_ols_matches_dummy_regression():
    rng = np.random.default_rng(1)
    n = 4000
    f1 = rng.integers(0, 100, n); f2 = rng.integers(0, 30, n)
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    a = rng.normal(size=100); b = rng.normal(size=30)
    y = 1.25*x1 - .7*x2 + a[f1] + b[f2] + rng.normal(scale=.2,size=n)
    df = pd.DataFrame(dict(y=y,x1=x1,x2=x2,f1=f1,f2=f2))
    got = reghdfe(df, y="y", x=["x1","x2"], absorb=["f1","f2"], vce="iid")
    ref = _dummy_ols(df,"y",["x1","x2"],["f1","f2"])
    np.testing.assert_allclose(got.params, ref, atol=2e-8, rtol=2e-8)
    assert got.converged


def test_iv_recovers_structural_coefficient():
    rng = np.random.default_rng(2)
    n=12000
    f=rng.integers(0,300,n); t=rng.integers(0,20,n)
    z=rng.normal(size=n); w=rng.normal(size=n); v=rng.normal(size=n)
    u=.6*v+rng.normal(size=n)
    x=.9*z+.4*w+v+rng.normal(scale=.2,size=n)
    af=rng.normal(size=300); at=rng.normal(size=20)
    y=2.0*x+.5*w+af[f]+at[t]+u
    df=pd.DataFrame(dict(y=y,x=x,z=z,w=w,f=f,t=t))
    got=ivreghdfe(df,y="y",exog=["w"],endog=["x"],instruments=["z"],absorb=["f","t"],vce="robust")
    assert abs(got.params[-1]-2.0) < .08
    assert got.converged


def test_singletons_are_iterated():
    df=pd.DataFrame({"y":[1.,2.,3.,4.,5.],"x":[1.,2.,0.,1.,2.],"a":[0,0,1,2,2],"b":[0,1,1,2,2]})
    got=reghdfe(df,y="y",x=["x"],absorb=["a","b"],drop_singletons=True)
    assert got.dropped_singletons >= 2

def test_iv_matches_explicit_dummy_2sls():
    rng=np.random.default_rng(7)
    n=1200
    f=rng.integers(0,35,n); t=rng.integers(0,8,n)
    z=rng.normal(size=n); w=rng.normal(size=n); v=rng.normal(size=n)
    x=.8*z+.3*w+v
    af=rng.normal(size=35); at=rng.normal(size=8)
    y=1.7*x-.4*w+af[f]+at[t]+.5*v+rng.normal(size=n)
    df=pd.DataFrame(dict(y=y,x=x,z=z,w=w,f=f,t=t))
    got=ivreghdfe(df,y="y",exog=["w"],endog=["x"],instruments=["z"],absorb=["f","t"],vce="iid")
    D=pd.get_dummies(df[["f","t"]].astype("category"),drop_first=False,dtype=float).to_numpy()
    X=np.column_stack([w,x,D]); Z=np.column_stack([w,z,D])
    PZX=Z @ (np.linalg.pinv(Z.T@Z) @ (Z.T@X))
    bref=np.linalg.pinv(X.T@PZX) @ (X.T @ (Z @ (np.linalg.pinv(Z.T@Z) @ (Z.T@y))))
    np.testing.assert_allclose(got.params,bref[:2],atol=1e-8,rtol=1e-8)


def test_oneway_cluster_vcov_matches_full_dummy_statsmodels():
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster
    rng=np.random.default_rng(11)
    n=2500
    f=rng.integers(0,70,n); t=rng.integers(0,12,n); c=rng.integers(0,40,n)
    x1=rng.normal(size=n); x2=rng.normal(size=n)
    y=.8*x1-.2*x2+rng.normal(size=70)[f]+rng.normal(size=12)[t]+rng.normal(size=n)
    df=pd.DataFrame(dict(y=y,x1=x1,x2=x2,f=f,t=t,c=c))
    got=reghdfe(df,y="y",x=["x1","x2"],absorb=["f","t"],cluster="c",vce="cluster")
    D=pd.get_dummies(df[["f","t"]].astype("category"),drop_first=False,dtype=float).to_numpy()
    A=np.column_stack([x1,x2,D])
    full=sm.OLS(y,A).fit()
    V0=cov_cluster(full,c,use_correction=False)
    # statsmodels CR1 counts raw columns even when the dummy matrix is singular;
    # pyreghdfe uses the actual rank, matching reghdfe-style redundant-FE logic.
    k_total=np.linalg.matrix_rank(A)
    scale=(len(np.unique(c))/(len(np.unique(c))-1))*((n-1)/(n-k_total))
    V=V0*scale
    np.testing.assert_allclose(got.params,full.params[:2],atol=1e-8,rtol=1e-8)
    np.testing.assert_allclose(got.vcov,V[:2,:2],atol=2e-8,rtol=2e-8)

def test_lsmr_matches_map():
    rng=np.random.default_rng(23)
    n=5000
    f=rng.integers(0,200,n); t=rng.integers(0,25,n)
    x=rng.normal(size=(n,3)); y=x@np.array([.2,-1.1,2.3])+rng.normal(size=200)[f]+rng.normal(size=25)[t]+rng.normal(size=n)
    a=reghdfe(None,y=y,x=x,absorb=[f,t],method="map",tol=1e-10)
    b=reghdfe(None,y=y,x=x,absorb=[f,t],method="lsmr",tol=1e-10)
    np.testing.assert_allclose(a.params,b.params,atol=2e-8,rtol=2e-8)
    assert b.converged

def test_parallel_wild_bootstrap_shape_and_reproducibility():
    from pyreghdfe import wild_bootstrap
    rng=np.random.default_rng(42)
    n=1800; f=rng.integers(0,60,n); t=rng.integers(0,9,n); c=rng.integers(0,30,n)
    X=rng.normal(size=(n,2)); y=X@np.array([.5,-.25])+rng.normal(size=60)[f]+rng.normal(size=9)[t]+rng.normal(size=n)
    r=reghdfe(None,y=y,x=X,absorb=[f,t],cluster=c,vce="cluster",keep_state=True)
    a=wild_bootstrap(r,reps=12,n_jobs=2,batch_size=3,seed=9)
    b=wild_bootstrap(r,reps=12,n_jobs=1,batch_size=3,seed=9)
    assert a.shape==(12,2)
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)

def test_cg_acceleration_matches_map():
    rng=np.random.default_rng(51)
    n=6000; f=rng.integers(0,250,n); t=rng.integers(0,35,n)
    X=rng.normal(size=(n,2)); y=X@np.array([1.2,-.8])+rng.normal(size=250)[f]+rng.normal(size=35)[t]+rng.normal(size=n)
    plain=reghdfe(None,y=y,x=X,absorb=[f,t],tol=1e-9,acceleration="none")
    cg=reghdfe(None,y=y,x=X,absorb=[f,t],tol=1e-9,acceleration="cg")
    np.testing.assert_allclose(plain.params,cg.params,atol=2e-8,rtol=2e-8)
    assert cg.converged

def test_weighted_ols_matches_weighted_dummy_regression():
    rng=np.random.default_rng(77)
    n=3000
    f=rng.integers(0,80,n); t=rng.integers(0,15,n)
    X=rng.normal(size=(n,2)); w=.2+rng.random(n)*2
    y=X@np.array([.7,-1.4])+rng.normal(size=80)[f]+rng.normal(size=15)[t]+rng.normal(size=n)
    got=reghdfe(None,y=y,x=X,absorb=[f,t],weights=w,vce="iid",tol=1e-10)
    D=pd.get_dummies(pd.DataFrame({"f":f,"t":t}).astype("category"),drop_first=False,dtype=float).to_numpy()
    A=np.column_stack([X,D]); sw=np.sqrt(w)
    bref,*_=np.linalg.lstsq(A*sw[:,None],y*sw,rcond=None)
    np.testing.assert_allclose(got.params,bref[:2],atol=3e-8,rtol=3e-8)


def test_heterogeneous_intercept_slope_matches_explicit_dummy():
    from pyreghdfe import FixedEffect
    rng = np.random.default_rng(101)
    n = 5000
    f = rng.integers(0, 90, n); t = rng.integers(0, 12, n)
    z = rng.normal(size=n); x = rng.normal(size=n)
    af = rng.normal(size=90); bf = rng.normal(scale=.4, size=90); at = rng.normal(size=12)
    y = 1.35*x + af[f] + bf[f]*z + at[t] + rng.normal(scale=.3, size=n)
    got = reghdfe(None, y=y, x=x, absorb=[FixedEffect(f, slopes=(z,), intercept=True), t], tol=1e-10)
    Df = pd.get_dummies(pd.Series(f).astype('category'), drop_first=False, dtype=float).to_numpy()
    Dt = pd.get_dummies(pd.Series(t).astype('category'), drop_first=False, dtype=float).to_numpy()
    A = np.column_stack([x, Df, Df*z[:,None], Dt])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    np.testing.assert_allclose(got.params, b[:1], atol=5e-8, rtol=5e-8)
    assert got.converged


def test_heterogeneous_slope_only_matches_explicit_dummy():
    from pyreghdfe import FixedEffect
    rng = np.random.default_rng(102)
    n = 3500
    f = rng.integers(0, 70, n); t = rng.integers(0, 10, n)
    z = rng.normal(size=n); x = rng.normal(size=n)
    bf = rng.normal(scale=.5, size=70); at = rng.normal(size=10)
    y = -.8*x + bf[f]*z + at[t] + rng.normal(scale=.2, size=n)
    got = reghdfe(None, y=y, x=x, absorb=[FixedEffect(f, slopes=(z,), intercept=False), t], tol=1e-10)
    Df = pd.get_dummies(pd.Series(f).astype('category'), drop_first=False, dtype=float).to_numpy()
    Dt = pd.get_dummies(pd.Series(t).astype('category'), drop_first=False, dtype=float).to_numpy()
    A = np.column_stack([x, Df*z[:,None], Dt])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    np.testing.assert_allclose(got.params, b[:1], atol=5e-8, rtol=5e-8)


def test_multiple_heterogeneous_slopes_and_lsmr_match_map():
    from pyreghdfe import FixedEffect
    rng = np.random.default_rng(103)
    n = 4500
    f = rng.integers(0, 80, n); t = rng.integers(0, 11, n)
    s1 = rng.normal(size=n); s2 = rng.normal(size=n); X = rng.normal(size=(n,2))
    a = rng.normal(size=80); b1 = rng.normal(size=80); b2 = rng.normal(size=80)
    y = X@np.array([.4,-1.2]) + a[f] + b1[f]*s1 + b2[f]*s2 + rng.normal(size=11)[t] + rng.normal(scale=.2,size=n)
    spec = [FixedEffect(f, slopes=(s1,s2), intercept=True), t]
    mapr = reghdfe(None,y=y,x=X,absorb=spec,method='map',tol=1e-9)
    lsmr = reghdfe(None,y=y,x=X,absorb=spec,method='lsmr',tol=1e-9,max_iter=5000)
    np.testing.assert_allclose(mapr.params, lsmr.params, atol=2e-7, rtol=2e-7)
    assert mapr.converged and lsmr.converged


def test_pairwise_dof_three_way_matches_reghdfe_rule():
    from pyreghdfe.dof import absorbed_dof, pairwise_components
    # Deliberately disconnected pair structures.
    a = np.array([0,0,1,1,2,2,3,3], dtype=np.int32)
    b = np.array([0,0,0,0,1,1,1,1], dtype=np.int32)
    c = np.array([0,1,0,1,2,3,2,3], dtype=np.int32)
    info = absorbed_dof([a,b,c], method='pairwise', adjust_nested=False)
    k1,k2,k3 = [len(np.unique(g)) for g in (a,b,c)]
    m2 = pairwise_components(a,b)
    m3 = max(pairwise_components(a,c), pairwise_components(b,c))
    assert info.df_absorbed == k1 + (k2-m2) + (k3-m3)
    assert info.m_by_component == (0,m2,m3)


def test_continuous_dof_detects_constant_slope_groups():
    from pyreghdfe.dof import absorbed_dof
    g = np.array([0,0,1,1,2,2,3,3], dtype=np.int32)
    # groups 0 and 2 are constant; groups 1 and 3 vary
    s = np.array([1,1,0,2,5,5,-1,1], dtype=float)[:,None]
    info = absorbed_dof([g], slopes=[s], intercepts=[True], adjust_nested=False)
    assert info.k_by_component == (4,4)
    assert info.m_by_component == (0,2)
    assert info.df_absorbed == 6


def _explicit_within(A, groups):
    D = pd.get_dummies(pd.DataFrame({f'f{i}':g for i,g in enumerate(groups)}).astype('category'), drop_first=False, dtype=float).to_numpy()
    coef, *_ = np.linalg.lstsq(D, A, rcond=None)
    return A - D@coef


def test_liml_matches_direct_kclass_formula_after_explicit_absorption():
    rng = np.random.default_rng(111)
    n=2200
    f1=rng.integers(0,45,n); f2=rng.integers(0,9,n)
    z1=rng.normal(size=n); z2=rng.normal(size=n); w=rng.normal(size=n); v=rng.normal(size=n)
    x=.45*z1+.30*z2+.2*w+v
    y=1.6*x+.3*w+rng.normal(size=45)[f1]+rng.normal(size=9)[f2]+.6*v+rng.normal(size=n)
    got=ivreghdfe(None,y=y,exog=w,endog=x,instruments=np.column_stack([z1,z2]),absorb=[f1,f2],estimator='liml',vce='iid',tol=1e-10)
    A=_explicit_within(np.column_stack([y,w,x,z1,z2]),[f1,f2])
    yy=A[:,0]; C=A[:,1:2]; E=A[:,2:3]; I=A[:,3:]
    X=np.column_stack([C,E]); Z=np.column_stack([C,I]); ee=np.column_stack([yy,E])
    ez=ee-Z@(np.linalg.pinv(Z)@ee)
    ex1=ee-C@(np.linalg.pinv(C)@ee)
    vals,vecs=np.linalg.eigh(ez.T@ez)
    invroot=(vecs/np.sqrt(vals))@vecs.T
    kap=np.min(np.linalg.eigvalsh(invroot@(ex1.T@ex1)@invroot))
    pinvz=np.linalg.pinv(Z)
    P1=(X.T@X)*(1-kap)+kap*((X.T@Z)@(pinvz@X))
    P2=(X.T@yy)*(1-kap)+kap*((X.T@Z)@(pinvz@yy))
    bref=np.linalg.solve(P1,P2)
    np.testing.assert_allclose(got.params,bref,atol=2e-7,rtol=2e-7)
    assert abs(got.kappa-kap) < 2e-7


def test_gmm2s_exact_identification_equals_2sls():
    rng=np.random.default_rng(112)
    n=5000
    f=rng.integers(0,100,n); t=rng.integers(0,15,n)
    z=rng.normal(size=n); w=rng.normal(size=n); v=rng.normal(size=n)
    x=.8*z+.2*w+v
    y=2.2*x-.25*w+rng.normal(size=100)[f]+rng.normal(size=15)[t]+.7*v+rng.normal(size=n)
    a=ivreghdfe(None,y=y,exog=w,endog=x,instruments=z,absorb=[f,t],estimator='2sls',vce='robust')
    b=ivreghdfe(None,y=y,exog=w,endog=x,instruments=z,absorb=[f,t],estimator='gmm2s',vce='robust')
    np.testing.assert_allclose(a.params,b.params,atol=2e-8,rtol=2e-8)
    assert b.diagnostics['overidentification']['df']==0


def test_iv_diagnostics_are_exposed():
    rng=np.random.default_rng(113)
    n=4000
    f=rng.integers(0,80,n); t=rng.integers(0,10,n)
    z1=rng.normal(size=n); z2=rng.normal(size=n); w=rng.normal(size=n); v=rng.normal(size=n)
    x=.7*z1+.4*z2+.2*w+v
    y=1.8*x+.1*w+rng.normal(size=80)[f]+rng.normal(size=10)[t]+.5*v+rng.normal(size=n)
    r=ivreghdfe(None,y=y,exog=w,endog=x,instruments=np.column_stack([z1,z2]),absorb=[f,t],vce='robust')
    fs=r.first_stage['diagnostics'][0]
    assert fs['partial_r2'] > .1
    assert fs['f_robust'] > 10
    assert np.isfinite(r.diagnostics['cragg_donald_f'])
    assert r.diagnostics['overidentification']['df'] == 1
