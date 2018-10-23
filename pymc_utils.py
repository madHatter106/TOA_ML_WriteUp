import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as pl
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class Ordered(pm.distributions.transforms.ElemwiseTransform):
    name = "ordered"

    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0], x[0])
        out = tt.inc_subtensor(out[1:], tt.log(x[1:] - x[:-1]))
        return out
    
    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0], y[0])
        out = tt.inc_subtensor(out[1:], tt.exp(y[1:]))
        return tt.cumsum(out)

    def jacobian_det(self, y):
        return tt.sum(y[1:])


class PyMCModel:
    def __init__(self, model, X, y, model_name='None', **model_kws):
        self.model = model(X, y, **model_kws)
        self.model.name = model_name
        
    def fit(self, n_samples=2000, **sample_kws):
        with self.model:
            self.trace_ = pm.sample(n_samples, **sample_kws)
    
    def fit_ADVI(self, n_samples=2000, n_iter=100000, inference='advi', **fit_kws):
        with self.model:
            self.approx_fit = pm.fit(n=n_iter, method=inference, **fit_kws)
            self.trace_ = self.approx_fit.sample(draws=n_samples)
    
    def show_model(self, save=False, view=True, cleanup=True):
        model_graph = pm.model_to_graphviz(self.model)
        if save:
            model_graph.render(save, view=view, cleanup=cleanup)
        if view:
            return model_graph
    
    def predict(self, likelihood_name='likelihood', **ppc_kws):
        ppc_ = pm.sample_ppc(self.trace_, model=self.model,
                             **ppc_kws)[likelihood_name]
        return ppc_
    
    def evaluate_fit(self, show_feats):
        return pm.traceplot(self.trace_, varnames=show_feats)
    
    def show_forest(self, show_feats, feat_labels=None):
        g = pm.forestplot(self.trace_, varnames=show_feats,
                             ylabels=feat_labels)
        f = pl.gcf()
        try:
            ax = f.get_axes()[1]
        except IndexError:
            ax = f.get_axes()[0]
        ax.grid(axis='y')
        return g
    
    
    def plot_model_ppc_stats(self, ppc, y_obs, alpha_level1=0.05,
                             alpha_level2=0.5, ax=None):
        if ax is None:
            _, ax = pl.subplots()
        iy = np.argsort(y_obs)
        ix = np.arange(iy.size)
        ppc_mean = ppc.mean(axis=0)
        ax.scatter(ix, y_obs.values[iy], label='observed', edgecolor='k', s=50,
                   color='steelblue')
        ax.scatter(ix, ppc_mean[iy], label='prediction mean', edgecolor='k', s=50,
                   color='red')
                 
        if alpha_level2:
            lik_hpd_2 = pm.hpd(ppc, alpha=alpha_level2)
            ax.fill_between(ix, y1=lik_hpd_2[iy, 0], y2=lik_hpd_2[iy, 1], alpha=0.5,
                            color='k',
                            label=f'prediction {1-alpha_level2:.2f}%CI',)
        if alpha_level1:
            lik_hpd_1 = pm.hpd(ppc, alpha=alpha_level1)
            ax.fill_between(ix, y1=lik_hpd_1[iy, 0], y2=lik_hpd_1[iy, 1], alpha=0.5,
                            color='k', label=f'prediction {1-alpha_level1:.2f}%CI',)
        ax.legend(loc='best')
        return ax
    
    def plot_model_fits2(self, y_obs, y_pred=None, title=None, ax=None, ci=0.95):
        if y_pred is None:
            y_pred = self.trace_.get_values('mu')
        y_obs = y_obs.values
        mask = np.logical_not(np.isnan(y_obs))
        y_obs = y_obs[mask]
        y_pred_mean = np.mean(y_pred, axis=0)[mask]
        y_pred_hpd = pm.hpd(y_pred, alpha=1-ci)[mask]
        xi = np.arange(y_obs.size)
        iy = np.argsort(y_obs)
        if ax is None:
            _, ax = pl.subplots(figsize=(12, 8),)
        ax.set_title(title)
        ax.plot(xi, y_obs[iy], marker='.', ls='',
                markeredgecolor='darkblue', markersize=13,
                label='observed')
        ax.plot(xi, y_pred_mean[iy], marker='o', color='indigo',
                ls='', markeredgecolor='k', alpha=0.5, label='predicted avg.')
        ax.fill_between(xi, y_pred_hpd[iy, 0], y_pred_hpd[iy, 1],
                        color='k', alpha=0.5,
                        label=f'{ci*100}%CI on pred.' );
        ax.legend(loc='best')
        return ax

    
def hs_regression(X, y_obs, ylabel='y', tau_0=None, regularized=False, **kwargs):
    """See Piironen & Vehtari, 2017 (DOI: 10.1214/17-EJS1337SI)"""
    X_ = pm.floatX(X)
    Y_ = pm.floatX(y_obs)
    n_features = X_.eval().shape[1]
    if tau_0 is None:
        m0 = n_features/2
        n_obs = X_.eval().shape[0]
        tau_0 = m0 / ((n_features - m0) * np.sqrt(n_obs))
    with pm.Model() as model:
        tau = pm.HalfCauchy('tau', tau_0)
        sd_bias = pm.HalfCauchy('sd_bias', beta=2.5)
        lamb_m = pm.HalfCauchy('lambda_m', beta=1)
        
    if regularized:
        slab_scale = kwargs.pop('slab_scale', 3)
        slab_scale_sq = slab_scale ** 2
        slab_df = kwargs.pop('slab_df', 8)
        half_slab_df = slab_df / 2
        with model:
            c_sq = pm.InverseGamma('c_sq', alpha=half_slab_df,
                                   beta=half_slab_df * slab_scale_sq)
            lamb_m_bar = tt.sqrt(c_sq) * lamb_m / (tt.sqrt(c_sq + 
                                                           tt.pow(tau, 2) *
                                                           tt.pow(lamb_m, 2)
                                                          )
                                                  )
            w = pm.Normal('w', mu=0, sd=tau*lamb_m_bar, shape=n_features)
    else:
        with model:
            w = pm.Normal('w', mu=0, sd=tau*lamb_m, shape=n_features)
            
    with model:
            bias = pm.Laplace('bias', mu=0, b=sd_bias)
            mu_ = pm.Deterministic('mu', tt.dot(X_, w) + bias)
            sig = pm.HalfCauchy('sigma', beta=5)
            y = pm.Normal(ylabel, mu=mu_, sd=sig, observed=Y_)
    return model

    
def lasso_regression(X, y_obs, ylabel='y'):
    num_obs, num_feats = X.eval().shape
    with pm.Model() as mlasso:
        sd_beta = pm.HalfCauchy('sd_beta', beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=0, b=sd_beta)
        w = pm.Laplace('w', mu=0, b=sd_beta, shape=num_feats)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X, w))
        y = pm.Normal(ylabel, mu=mu_, sd=sig, observed=y_obs.squeeze())
    return mlasso


def lasso_regr_impute_y(X, y_obs, ylabel='y'):
    num_obs, num_feats = X.eval().shape
    with pm.Model() as mlass_y_na:
        sd_beta = pm.HalfCauchy('sd_beta', beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=0, b=sd_beta)
        w = pm.Laplace('w', mu=0, b=sd_beta, shape=num_feats)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X, w))
        mu_y_obs = pm.Normal('mu_y_obs', 0.5, 1)
        sigma_y_obs = pm.HalfCauchy('sigma_y_obs', 1)
        y_obs_ = pm.Normal('y_obs', mu_y_obs, sigma_y_obs, observed=y_obs.squeeze())
        y = pm.Normal(ylabel, mu=y_obs_, sd=sig)
    return mlass_y_na


def hier_lasso_regr(X, y_obs, add_bias=True, ylabel='y'):
    X_ = pm.floatX(X)
    Y_ = pm.floatX(y_obs)
    n_features = X_.eval().shape[1]
    with pm.Model() as mlasso:
        hyp_beta = pm.HalfCauchy('hyp_beta', beta=2.5)
        hyp_mu = pm.HalfCauchy('hyp_mu', mu=0, beta=2.5)
        sig = pm.HalfCauchy('sigma', beta=2.5)
        bias = pm.Laplace('bias', mu=hyp_mu, b=hyp_beta)
        w = pm.Laplace('w', mu=hyp_mu, b=hyp_beta, shape=n_features)
        mu_ = pm.Deterministic('mu', bias + tt.dot(X_, w))
        y = pm.Normal(ylabel, mu=mu_, sd=sig, observed=Y_)
    return mlasso


def partial_pooling_lasso(X, y_obs, ylabel='y'):
    pass


def plot_fits_with_unc(y_obs, ppc_, ax=None):
    iy  = np.argsort(y_obs)
    ix = np.arange(iy.size)
    lik_mean =ppc_.mean(axis=0)
    lik_hpd = pm.hpd(ppc_)
    lik_hpd_05 = pm.hpd(ppc_, alpha=0.5)
    if ax is None:
        _, ax = pl.subplots(figsize=(12, 8))
        ax.scatter(ix, y_obs.values[iy], label='observed', edgecolor='k', s=50,
                   color='steelblue');
        ax.scatter(ix, lik_mean[iy], label='modeled', edgecolor='k', s=50, color='m')

        ax.fill_between(ix, y1=lik_hpd_05[iy, 0], y2=lik_hpd_05[iy, 1], alpha=0.5, color='k',
                       label='model output 50%CI');
        ax.fill_between(ix, y1=lik_hpd[iy, 0], y2=lik_hpd[iy, 1], alpha=0.5, color='k',
                       label='model output 95%CI');
        ax.legend(loc='upper left');
    return ax


def subset_significant_feature(trace, labels_list, alpha=0.05, vars_=None):
    if vars_ is None:
        vars_ = ['sd_beta', 'sigma', 'bias', 'w']
    dsum = pm.summary(trace, varnames=vars_, alpha=alpha)
    lbls_list = ['w[%s]' %lbl for lbl in labels_list]
    dsum.index = vars_[:-1] + lbls_list 
    hpd_lo, hpd_hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    if str(hpd_lo).split('.')[1] == '0':
        hpd_lo = int(hpd_lo)
    if str(hpd_hi).split('.')[1] == '0':
        hpd_hi = int(hpd_hi)
    dsum_subset = dsum[(((dsum[f'hpd_{hpd_lo}']<0)&(dsum[f'hpd_{hpd_hi}']<0))|
                    ((dsum[f'hpd_{hpd_lo}']>0) & (dsum[f'hpd_{hpd_hi}']>0))
                   )]
    pattern1 = r'w\s*\[([a-z_\sA-Z0-9]+)\]'
    return list(dsum_subset.index.str.extract(pattern1).dropna().values.flatten())


def plot_pairwise_corr(df_, ax=None):
    if ax is None:
        _, ax = pl.subplots(figsize=(12, 10))
    heatmap(df_.corr().iloc[1:,:-1],vmin=-1, vmax=1,
            mask=np.triu(np.ones([df_.shape[1]-1] * 2),k=1),
            ax=ax, annot=True, annot_kws={'fontsize': 10})
    ax.set_facecolor('k')
    return ax


def plot_fits_w_estimates(y_obs, ppc, ax=None, savename=False):
    """ Plot Fits with Uncertainty Estimates"""
    iy  = np.argsort(y_obs)
    ix = np.arange(iy.size)
    lik_mean =ppc.mean(axis=0)
    lik_hpd = pm.hpd(ppc)
    lik_hpd_05 = pm.hpd(ppc, alpha=0.5)
    if ax is None:
        _, ax = pl.subplots(figsize=(12, 8))
    ax.scatter(ix, y_obs.values[iy], label='observed', edgecolor='k', s=100,
               color='steelblue', marker='d', zorder=2);
    ax.scatter(ix, lik_mean[iy], label='modeled', edgecolor='k', s=100, color='orange', zorder=3)

    ax.fill_between(ix, y1=lik_hpd_05[iy, 0], y2=lik_hpd_05[iy, 1], color='gray', 
                   label='model output 50%CI', zorder=1,linestyle='-', lw=2, edgecolor='k');
    ax.fill_between(ix, y1=lik_hpd[iy, 0], y2=lik_hpd[iy, 1], color='k', alpha=0.75,
                   label='model output 95%CI', zorder=0, );
    ax.legend(loc='upper left');
    if savename:
        f = pl.gcf()
        f.savefig('./figJar/bayesNet/%s.pdf' % savename, format='pdf')
    return ax


def evaluate_model(model,  y_train_, y_test_, ax1_title=None, ax2_title=None, ax3_title=None,):
    """Makes a 3-way panel to evaluate model w/ training and testing"""
    f = pl.figure(figsize=(15, 15))
    ax1 = pl.subplot2grid((2, 2), (0, 0))
    ax2 = pl.subplot2grid((2, 2), (0, 1))
    ax3 = pl.subplot2grid((2, 2), (1, 0), colspan=2)
    X_shared.set_value(X_s_train)
    ppc_train_ = model.predict(likelihood_name='likelihood' )
    model.plot_model_fits(y_train_, ppc_train_, loss_metric='mae',
                          ax=ax1, title=ax1_title, );
    X_shared.set_value(X_s_test)
    ppc_test_ = model.predict()
    model.plot_model_fits(y_test_, ppc_test_, loss_metric='mae',
                          ax=ax2, title=ax2_title, );
    plot_fits_w_estimates(y_test.log10_aphy411, ppc_test_411, ax=ax3)
    ax3.set_title(ax3_title)
    return f