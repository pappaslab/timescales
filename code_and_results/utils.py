
def reorder_BNA_ctx_ROIs(data, direction="surf_to_vol"):
    import numpy as np

    if direction == 'surf_to_vol':
        data_ordered = np.zeros((210,))
        data_ordered[::2] = data[105:] # Left hemisphere
        data_ordered[1::2] = data[:105] # Right hemisphere
    elif direction == 'vol_to_surf':
        data_ordered = np.zeros((210,))
        data_ordered[105:] = data[::2] # Left hemisphere
        data_ordered[:105] = data[1::2] # Right hemisphere
    
    return data_ordered
    
def subset_ROIs(data, atlas="BNA", hemi="both", division="ctx", symmetric=False):
    import warnings
    # ROIs must be the first dimension of data.
    if atlas == 'LG400':
        # Subset hemispheres.
        if  hemi == 'both':
            data_subset = data
            warnings.warn("No subset selected; all ROIs will be returned.")
        else:
            if symmetric == True:
                if hemi == 'left':
                    data_subset = data[tuple(data.ndim * [slice(200)])]
                elif hemi == 'right':
                    data_subset = data[tuple(data.ndim * [slice(200,None)])]
            else:
                if hemi == 'left':
                    data_subset = data[:200]
                elif hemi == 'right':
                    data_subset = data[200:]
        return data_subset
    elif atlas == 'BNA':
         # Warn if the subset includes all ROIs
        if hemi == 'both' and division == 'both':
            warnings.warn("No subset selected; all ROIs will be returned.")
        else:
            if division == 'both':
                data_subset = data
            else: 
                # Subset cortex vs subcortex
                if symmetric == True:
                    if division== 'ctx':
                        data_subset = data[tuple(data.ndim * [slice(210)])]
                    elif division == 'subctx':
                        data_subset = data[tuple(data.ndim * [slice(210,None)])]
                else: 
                    if division== 'ctx':
                        data_subset = data[:210]
                    elif division == 'subctx':
                        data_subset = data[210:]
            # Subset hemispheres
            if hemi == 'both':
                data_subset = data_subset
            else:
                if symmetric == True:
                    if hemi == 'left':
                        data_subset = data_subset[tuple(data.ndim * [slice(0,None,2)])]
                    elif hemi == 'right':
                        data_subset = data_subset[tuple(data.ndim * [slice(1,None,2)])]
                else:
                    if hemi == 'left':
                        data_subset = data_subset[::2]
                    elif hemi == 'right':
                        data_subset = data_subset[1::2]
       
    return data_subset

def join_hemispheres(lh_data, rh_data, atlas="BNA"):
    import numpy as np
    # lh_data and rh_data must be the same shape and either 1D or 2D.

    if len(lh_data.shape) == 1:
        lh_data = np.expand_dims(lh_data, axis=1)
        rh_data = np.expand_dims(rh_data, axis=1)
        axis2_dim = 1
    else:
        axis2_dim = lh_data.shape[0]
        lh_data = lh_data.T
        rh_data = rh_data.T
    

    if atlas == 'LG400':
        data_all = np.zeros((400,axis2_dim))
        data_all[:200] = lh_data
        data_all[200:] = rh_data
    elif atlas == 'BNA':
        data_all = np.zeros((lh_data.shape[0]*2,axis2_dim))
        data_all[::2] = lh_data
        data_all[1::2] = rh_data

    if axis2_dim == 1:
        data_all = np.squeeze(data_all)
    else:
        data_all = data_all.T

    return data_all

def join_BNA_divisions(ctx_data, subctx_data):
    import numpy as np

    if len(ctx_data.shape) == 1:
        ctx_data = np.expand_dims(ctx_data, axis=1)
        subctx_data = np.expand_dims(subctx_data, axis=1)
        axis2_dim = 1
    else:
        axis2_dim = ctx_data.shape[0]
        ctx_data = ctx_data.T
        subctx_data = subctx_data.T
    
    data_all = np.zeros((246,axis2_dim))
    data_all[:210] = ctx_data
    data_all[210:] = subctx_data

    if axis2_dim == 1:
        data_all = np.squeeze(data_all)
    else:
        data_all = data_all.T

    return data_all

def gen_surrogates(data, dist_mat, n_perms=10000, resample=False):
    from brainsmash.mapgen.base import Base

    base = Base(x=data, D=dist_mat, resample=resample, seed=247)
    surrogates = base(n=n_perms)

    return surrogates
        
def perm_helper(data, n_perms, dist_mat, atlas='BNA', split_by='division'):
## coord, atlas, etc are all passed through as perm_func_args
## split by: division (ctx vs subcortex), hemi (L/R), both (split cortex by hemi, split ctx from subctx)
# if split, then permute the splits separately before re-combining
# need to write tests to ensure appropriate inputs for each split case
## if splitby both, atlas must be BNA
## if splitby division, atlas must be BNA

    if split_by == None:
        surrogates = gen_surrogates(data, dist_mat, n_perms=n_perms)

    if split_by == 'hemi':
        # Pepare left cortex
        data_lh = subset_ROIs(data, atlas, hemi='left', division='both')
        dist_mat_lh = subset_ROIs(dist_mat, atlas, hemi='left', division='both', symmetric=True)
        # Pepare right cortex
        data_rh = subset_ROIs(data, atlas, hemi='right', division='both')
        dist_mat_rh = subset_ROIs(dist_mat, atlas, hemi='right', division='both', symmetric=True)
        # Generate surrogates
        surrogates_lh = gen_surrogates(data_lh, dist_mat_lh, n_perms=n_perms)
        surrogates_rh = gen_surrogates(data_rh, dist_mat_rh, n_perms=n_perms)
        # Join surrogates
        surrogates = join_hemispheres(surrogates_lh, surrogates_rh, atlas="BNA")
    
    if split_by == 'division':
        # Pepare cortex
        data_ctx = subset_ROIs(data, atlas, hemi='both', division='ctx')
        dist_mat_ctx = subset_ROIs(dist_mat, atlas, hemi='both', division='ctx', symmetric=True)
        # Prepare subcortex
        data_subctx = subset_ROIs(data, atlas, hemi='both', division='subctx')
        dist_mat_subctx = subset_ROIs(dist_mat, atlas, hemi='both', division='subctx', symmetric=True)
        # Generate surrogates
        surrogates_ctx = gen_surrogates(data_ctx, dist_mat_ctx, n_perms=n_perms)
        surrogates_subctx = gen_surrogates(data_subctx, dist_mat_subctx, n_perms=n_perms)
        # Join surrogates
        surrogates = join_BNA_divisions(surrogates_ctx, surrogates_subctx)

    if split_by == 'both':
        # Pepare left cortex
        data_lh = subset_ROIs(data, atlas, hemi='left', division='ctx')
        dist_mat_lh = subset_ROIs(dist_mat, atlas, hemi='left', division='ctx', symmetric=True)
        # Pepare right cortex
        data_rh = subset_ROIs(data, atlas, hemi='right', division='ctx')
        dist_mat_rh = subset_ROIs(dist_mat, atlas, hemi='right', division='ctx', symmetric=True)
        # Prepare subcortex
        data_subctx = subset_ROIs(data, atlas, hemi='both', division='subctx')
        dist_mat_subctx = subset_ROIs(dist_mat, atlas, hemi='both', division='subctx', symmetric=True)
        # Generate surrogates
        surrogates_lh = gen_surrogates(data_lh, dist_mat_lh, n_perms=n_perms)
        surrogates_rh = gen_surrogates(data_rh, dist_mat_rh, n_perms=n_perms)
        surrogates_subctx = gen_surrogates(data_subctx, dist_mat_subctx, n_perms=n_perms)
        # Join surrogates
        surrogates_ctx = join_hemispheres(surrogates_lh, surrogates_rh, atlas="BNA")
        surrogates = join_BNA_divisions(surrogates_ctx, surrogates_subctx)
    
    return surrogates

def triu_to_mat(shape, triu):
    """
    Create a new symmetric matrix from a vector of off-diagonal upper triangle values.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the new array.
    triu : array_like
        Vector of off-diagonal upper triangle values.
        
    Returns:
    ________
    new_mat : array_like
        Symmetric matrix, with zeros on the diagonal and triu above and below the diagonal.    
    """
    import numpy as np
    
    new_mat = np.zeros(shape)
    new_mat[np.triu_indices_from(new_mat, k=1)] = triu
    new_mat[np.tril_indices_from(new_mat, k=-1)] = new_mat.T[np.tril_indices_from(new_mat, k=-1)]
    
    return new_mat

def load_cartography(data_dir, ref_part=None):
    import numpy as np
    import pandas as pd

    gamma_range = np.arange(5,36,1) / 10
    pc_store = []
    wd_store = []
    for y in gamma_range:
        if ref_part:
            pc_data = np.genfromtxt(f'{data_dir}/{ref_part}_gamma_{y}_PC.txt')
            wd_data = np.genfromtxt(f'{data_dir}/{ref_part}_gamma_{y}_WMDz.txt')
        else:
            pc_data = np.genfromtxt(f'{data_dir}/gamma_{y}_PC.txt')
            wd_data = np.genfromtxt(f'{data_dir}/gamma_{y}_WMDz.txt')
        pc_store.append(pc_data)
        wd_store.append(wd_data)
    df_idx = [str(i) for i in gamma_range]
    pc_df = pd.DataFrame(pc_store, index=df_idx).T
    wd_df = pd.DataFrame(wd_store, index=df_idx).T
    
    return pc_df, wd_df

def load_partitions(data_dir, ref_part=None):
    import numpy as np
    import pandas as pd

    gamma_range = np.arange(5,36,1) / 10
    part_store = []
    for y in gamma_range:
        part_data = np.genfromtxt(f'{data_dir}/gamma_{y}_GraphPartition.txt')
        part_store.append(part_data)
    df_idx = [str(i) for i in gamma_range]
    part_df = pd.DataFrame(part_store, index=df_idx).T
    
    return part_df

def plot_null_hist(r_val, r_null, xlab="|r|"):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats

    sns.set_style('ticks')
    r_null = np.abs(r_null)
    r_val = np.abs(r_val)
    f = sns.displot(r_null, aspect=2)
    f = plt.axvline(stats.scoreatpercentile(r_null, 95), ls='--', label='p = 0.05', c='darkorange')
    f = plt.axvline(r_val, c='tab:olive')
    f = plt.xlabel(xlab)
    f = plt.legend()

def get_YData(y_label, z_vars, data_df, results_dir, fmri_data_label, roi_subset):
    from collections import namedtuple
    import numpy as np
    import statsmodels.formula.api as smf
    
    YData = namedtuple('YData', ['y_raw', 'y_perms'])
    
    formula = f'lag1 ~ {z_vars}'
    resid = smf.ols(formula, data=data_df).fit().resid.values
    yperms = np.load(f'{results_dir}/stats/connectivity_preference/{fmri_data_label}_BNA_{roi_subset}_{y_label}_resid_yperms.npy')

    return YData(resid, yperms)

def calc_pcorrs(df, var_cols):
    import pandas as pd
    import pingouin as pg
    import numpy as np

    df_sub = df[var_cols]
    pcorr_df = pd.DataFrame(np.zeros((df_sub.shape[1], df_sub.shape[1])), index=df_sub.columns, columns=df_sub.columns)
    corr_cols = var_cols[:-1]
    for i in corr_cols:
        for j in corr_cols:
            if i == j:
                p_corr = 1
            else:
                p_corr = pg.partial_corr(data=df, x=i, y=j, covar='roi_vols').r.values[0]
            pcorr_df.loc[i, j] = p_corr
            
    return pcorr_df.iloc[:-1,:-1]

def subset_diffs(rdiff_df, sig_df, subset, re_cols):
    import pandas as pd

    if type(subset) == list:
        var_df = pd.DataFrame(rdiff_df[subset])
        annot_df = pd.DataFrame(sig_df[subset])
    elif type(subset) == str:
        var_df = rdiff_df[rdiff_df.columns[[subset in i for i in rdiff_df.columns]]]
        annot_df = sig_df[sig_df.columns[[subset in i for i in sig_df.columns]]]
    var_df = var_df.T[re_cols]
    annot_df = annot_df.T[re_cols]

    return var_df, annot_df

def plot_diffs(var_df, annot_df, height=5, splits=None, cmap='cividis', norm=None, cbar_label='Difference in Effect Size', discrete=False): 
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax1 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(20,height))
    sns.heatmap(var_df, cbar=True, linewidths=1, cmap=cmap, linecolor='dimgrey',
                ax=ax1, annot=annot_df, fmt='', annot_kws={'fontweight':'black', 'fontsize':'large'},
                cbar_kws={'label':cbar_label}, **{'norm':norm})
    colorbar = ax1.collections[0].colorbar
    colorbar.ax.yaxis.set_label_position('left')
    if splits:
        for vidx in splits:
            ax1.axvline(vidx, c='k', lw=7, ls='-')
            ax1.axvline(vidx, c='w', lw=5, ls='-')
    if discrete == True:
        colorbar = ax1.collections[0].colorbar
        colorbar.set_ticklabels(['1', '2', '3','4','5'])
        colorbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    plt.yticks(rotation=0)
    plt.tight_layout()

def gen_pairwise_CIs(pwc_vars, pcorr_df, N, crit_z, absval=False): 
    import sys
    import pandas as pd
    import numpy as np
    sys.path.append('/home/despo/dlurie/Projects/PyPALM/')
    import pypalm as pm

    n_vars = len(pwc_vars)
    ciz_df = pd.DataFrame(np.zeros((n_vars,n_vars)), index=pwc_vars, columns=pwc_vars)
    rdiff_df = pd.DataFrame(np.zeros((n_vars,n_vars)), index=pwc_vars, columns=pwc_vars)

    for i in pwc_vars:
        r_12 = pcorr_df.loc['lag1',i]
        for j in pwc_vars:
            r_13 = pcorr_df.loc['lag1',j]
            r_23 = pcorr_df.loc[i,j]
            if i == j:
                incl_zero = np.nan
            else:
                if absval == True:
                    r_12 = np.abs(r_12)
                    r_13 = np.abs(r_13)
                L, U = pm.zou2007_DOL(r_12, r_13, r_23, N, N, crit_z, crit_z)
                incl_zero = np.prod(np.array([L, U])) < 0

            ciz_df.loc[i,j] = incl_zero
            rdiff_df.loc[i,j] = np.abs(r_12 - r_13)
    
    return ciz_df, rdiff_df

def sig_to_annot(df):
    sig_df = df.replace(False, '*')
    sig_df = sig_df.replace(True, '')
    return sig_df

def get_consensus_diffs(sig_files):
    import pandas as pd
    import numpy as np

    df_store = []
    for file in sig_files:
        df_tmp = pd.read_csv(file, index_col=0)
        df_tmp = df_tmp.replace(np.nan, 0)
        df_tmp = df_tmp.replace('*', 1)
        df_store.append(df_tmp)
    df_stack = np.array(df_store)

    sum_df = pd.DataFrame(np.sum(df_stack, axis=0), index=df_tmp.index, columns=df_tmp.columns)
    sum_df[sum_df < 1] = np.nan

    rep_df = sum_df == 5
    rep_df = rep_df.replace(True, '*')
    rep_df = rep_df.replace(False, '')

    return sum_df, rep_df

def run_cartography_tests(gamma_range, dat_df, cart_df, y_var, x_var, z_var, surrogates):
    import sys
    import pandas as pd
    sys.path.append('/home/despo/dlurie/Projects/PyPALM/')
    import pypalm as pm

    r_store = []
    pgt_store = []
    plt_store = []
    pab_store = []

    for y in gamma_range:
        dat_tmp = dat_df.copy()
        dat_tmp[x_var] = cart_df[str(y)]
        
        rval, pvals, model = pm.freedman_lane(dat_tmp, y_var, x_var, z_var, 
                                              stat='pcorr', n_perms=10000, perm_func=None,
                                              perm_func_args=None, surrogates=surrogates,
                                              return_null=False, return_surrogates=False)
        r_store.append(rval)
        pgt_store.append(pvals.p_greater)
        plt_store.append(pvals.p_less)
        pab_store.append(pvals.p_abs)

    out_df = pd.DataFrame([r_store, pgt_store, plt_store, pab_store], 
                                    index = ['r_val', 'p_greater', 'p_less', 'p_abs'],
                                    columns=[str(i) for i in gamma_range])

    out_df = out_df.T
    out_df.index.name = 'gamma'
    out_df = out_df.reset_index()

    return out_df

def prep_summary_plots(q_df, r_df, other_index):
    import numpy as np
    import pandas as pd

    n_other = len(other_index)
    gamma_range = np.arange(5,36,1) / 10

    annot_df = q_df <= 0.05

    annot_df = annot_df.replace(True, '*')
    annot_df = annot_df.replace(False, '')

    r_df_cart = pd.DataFrame(r_df[:-n_other].values.reshape((4,31)), columns=gamma_range,
                                       index=['SC PC','FC PC','SC WD','FC WD'])

    annot_df_cart = annot_df[:-n_other].values.reshape((4,31))

    r_df_other = r_df[-n_other:].copy()
    r_df_other.name = 'q-value'
    r_df_other.index = other_index
    r_df_other = pd.DataFrame(r_df_other)
    
    annot_df_other = annot_df[-n_other:].values.reshape((n_other,1))
    
    return annot_df, r_df_cart, annot_df_cart, r_df_other, annot_df_other

def plot_cart_test(gamma_series, r_series, p_series):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('darkgrid', {'axes.linewidth':1, 'axes.edgecolor':'black'})
    fig, ax1 = plt.subplots(figsize=(14,5.5))
    ax1.scatter(gamma_series, r_series, label='$\mathit{{ρ}}_\mathit{{XY•Z}}$', color='#1f77b4')
    ax2 = ax1.twinx()
    ax2.scatter(gamma_series, p_series, color='orangered',  marker="*", alpha=0.33)
    ax2.scatter(gamma_series[p_series <= 0.05], p_series[p_series <= 0.05], label="p-value", color='orangered',  marker="*")
    ax1.set_xlabel('Resolution')
    ax1.set_ylabel('Partial Correlation')
    ax2.set_ylabel('$\mathit{{p}}_\mathit{{perm}}$')
    ax1.spines['left'].set_color('#1f77b4')
    ax2.spines['left'].set_color('#1f77b4')
    ax2.spines['right'].set_color('orangered')
    ax2.spines['right'].set_linewidth(3)
    ax2.spines['left'].set_linewidth(3)
    ax1.xaxis.grid(True)
    ax2.yaxis.grid(False)
    ax1.yaxis.grid(False)
    #fig.legend(loc='center right', bbox_to_anchor=(0.9,0.5))
    sns.despine(top=True, bottom=True, right=False)
    plt.tight_layout()

def plot_FDR_diagnostic(pvals, qvals):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    sns.set_context('notebook')
    sns.set_style('ticks')

    fig, ax1 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(6,6))
    ax1.scatter(pvals, qvals, marker='+')
    lin = (np.array(list(range(10)))/10)
    ax1.plot(lin, lin, ls=':', c='dimgrey')
    plt.axhline(0.05, ls='--', c='dimgrey')
    plt.axvline(0.05, ls='--', c='dimgrey')
    plt.xlim(0,.5)
    plt.ylim(0,.5)
    plt.xlabel('p-value (uncorrected)')
    plt.ylabel('q-value')
    plt.title('FDR Diagnostic')
    p_rej = np.sum(pvals <= 0.05)
    q_rej = np.sum(qvals <= 0.05)
    print(f'#H0 rejected based on uncorrected p-values: {p_rej}\n#H0 rejected based on FDR corrected p-values: {q_rej}\n')
    plt.tight_layout()

def plot_ta_partial(x, xlabel, y, ylabel, covar, rval, pval, qval, color, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_context('poster')
    sns.set_style('ticks')

    fig, ax1 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8,8))
    sns.regplot(x=x, y=y, x_partial=covar, y_partial=covar, color=color, ci=None,
                scatter_kws={'s':40, 'alpha':1}, line_kws={'lw':4})
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    texstring = r'$\mathit{{r}}$={0:04.3f}, $\mathit{{p}}_\mathit{{perm}}$={1:04.4f}, $\mathit{{p}}_\mathit{{perm^{{FDR}}}}$={2:04.4f}'.format(rval, pval, qval)
    ax1.text(.99,1.03, texstring,horizontalalignment='right', fontsize=18, transform=ax1.transAxes)
    ax1.tick_params(labelsize=16, axis='both')
    plt.title(title,y=1.13)
    plt.tight_layout()