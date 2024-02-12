def calc_diffs(feat_array, affinity_array=None, mask=None):
    """
    Calculate pairwise differences between elements of feat_array.
    
    Parameters
    ----------
    feat_array : array_like
        Array of shape (n_regions,) containing one feature value for each region
    affinity_array : array_like, optional
        Array of shape (n_regions, n_regions) containing standardized (ranging 0 to 1) pairwise affinities (e.g. correlations) between regions.

    Returns
    -------
    raw_diff_triu : array_like
        Upper triangle (off diagonal) of raw pairwise differences between elements of feat_array.
    norm_diff_triu : array_like
        If affinity_array is provided, the upper triangle (off diagonal) of pairwise differences between elements of feat_array, normalized by the affinity between the pair.
    """
    import numpy as np
    
    # Get pairwise differences between elements
    raw_diffs = np.abs(np.subtract.outer(feat_array, feat_array))
    # Grab just the upper triangle of the raw differences (keep only unique values)
    raw_diffs_triu = raw_diffs[np.triu_indices(raw_diffs.shape[0], k = 1)]

    # If an affinity matrix is provided...
    if affinity_array is not None:
        # Normalize pairwise differences by pairwise affinities
        norm_diffs = raw_diffs * affinity_array
        # Grab just the upper triangle of the normalized differences (keep only unique values)
        norm_diffs_triu = norm_diffs[np.triu_indices(norm_diffs.shape[0], k = 1)]

        return raw_diffs_triu, norm_diffs_triu
    else:
        return raw_diffs_triu
    
def null_corr(feat_array, affinity_array, n_perms=10000, feat_perms=None, dropz=True, perm_func=None, perm_func_args=None,
            return_surrogates=False, return_null=False):
    """
    Use permutations to test the hypothesis that the correlation between pairwise differences and affinities is stronger than expected by chance.
    
    Parameters
    ----------
    feat_array : array_like
        Array of shape (n_regions,) containing one feature value for each region
    affinity_array : array_like
        Array of shape (n_regions, n_regions) containing standardized (ranging 0 to 1) pairwise affinities (e.g. correlations) between regions.
    n_perms : int
        Number of null correlations to generate.
    feat_perms : array_like
        Array of shape (n_perms, n_regions) containing pre-shuffled feature values.
    dropz : bool
        Ignore pairs with an affinity of zero.
        
    Returns
    -------
    observed_corr : DataFrame
        Observed correlation between pairwise differences and pairwise affinities.
    null_corrs : array_like
        Distribution of null Pearson correlations generated by shuffling element identities.
    empirical_p : float
        Empirical p-value for the hypothesis that the observed correlation is significantly different from the distribution of null correlations.
    """
    import numpy as np
    import pandas as pd
    import pingouin as pg

    # Create a class to store outputs
    class ResultsStore:
        r_obs = None
        r_null = None
        p_vals = None
        surr = None
    Results = ResultsStore()
    
    # Grab the upper triangle of the affinity matrix
    affinities = affinity_array[np.triu_indices(affinity_array.shape[0], k = 1)]

    # Calcualte the observed differences.
    observed_diffs = calc_diffs(feat_array)
    
    ## If dropz is True, ignore pairs with an affinity of 0
    if dropz:
        zidx = affinities == 0
        observed_diffs = observed_diffs[~zidx]
        affinities = affinities[~zidx]
    
    ## Calculate and report observed correlation
    empirical_corr = pg.corr(observed_diffs, affinities, method='pearson')
    
    # If the user has provided previously generated surrogates for feat_array, use those.
    if feat_perms is not None:
        pass
    # Otherwise, generate surrogate data sets.
    # TO-DO: Generate permutations in parallel via multiprocessing/joblib/ray.
    else:
        # Permute the feature array.
        # If a custom permutation function is provided, use that.
        if perm_func:
            feat_perms = perm_func(feat_array, n_perms, **perm_func_args)
        # Otherwise, just do random shuffling.
        else:
            feat_perms = []
            for i in range(n_perms):
                feat_perms.append(np.random.permutation(feat_array))
            feat_perms = np.array(feat_perms)
    
    ## Estimate null distribution of correlations
    null_corrs = []
    for perm_iter in feat_perms:
        null_diffs, tmp = calc_diffs(perm_iter, affinity_array)
        ## If dropz is True, ignore pairs with an affinity of 0
        if dropz:
            null_diffs = null_diffs[~zidx]
        null_corrs.append(pg.corr(null_diffs, affinities).r.values[0])
    null_corrs = np.array(null_corrs)

    ## Calculate the empirical p-value based on permutations
    stat_observed = empirical_corr.r.values[0]
    p_greater = (np.sum(null_corrs >= stat_observed) + 1) / (n_perms + 1)
    p_less = (np.sum(null_corrs <= stat_observed) + 1) / (n_perms + 1)
    p_abs = (np.sum(np.abs(null_corrs) >= np.abs(stat_observed)) + 1) / (n_perms + 1)
    pvals = pd.Series([p_greater, p_less, p_abs], index=['p_greater', 'p_less', 'p_abs'])
    
    # Collect outputs.
    Results.r_obs = empirical_corr
    Results.p_vals = pvals
    if return_null:
        Results.r_null = null_corrs
    if return_surrogates:
        Results.surr = feat_perms 

    return Results

def binary_connection_test(feat_array, feat_surrogates, ed_array, affinity_array, return_null=True, return_model=False):
    """
    Test whether pairwise features similarities are for connected regions in a binarized network.
    
    Parameters
    ----------
    feat_array : array-like
        Feature to test within/between preference for
    feat_surrogates : array_like
        Precomputed surrogates for feat_array
    ed_array : array_like
        Pairwise distances between elements in feat_array
    affinity_array : array_like
        Connectivity matrix
    Returns
    -------
    e_T : float
        Observed t-value for the effect of connectivity.
    T_null : array-like
        Null distribution of t-values.
    p_vals : Series
        p-values for the observed effect relative to the null.
    e_OLfit: class
        Statsmodels model for the empirical data.

    Notes
    -----
    Setting cov_type='HC2' is equivalent to running a Welch's t-test which does not assume equal variances.
    This is necessary because the number of connections within vs. between differ greatly.
    """
    import bct
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm

    # Create a class to store outputs
    class ResultsStore:
        t_obs = None
        t_null = None
        p_vals = None
        model = None
    Results = ResultsStore()

    con = bct.degree.binarize(affinity_array)[np.triu_indices_from(affinity_array, k=1)]

    e_diffs = calc_diffs(feat_array)
    e_OLfit = sm.OLS(e_diffs, np.array([np.ones_like(e_diffs), con, ed_array]).T).fit(cov_type='HC2')
    e_T = e_OLfit.tvalues[1]

    T_null = []
    for surr in feat_surrogates:
        n_diffs = calc_diffs(surr)
        n_OLfit = sm.OLS(n_diffs, np.array([np.ones_like(n_diffs), con, ed_array]).T).fit(cov_type='HC2')        
        T_null.append(n_OLfit.tvalues[1])
    T_null = np.array(T_null)

    ## Calculate the empirical p-value based on permutations
    stat_observed = e_T
    n_perms = feat_surrogates.shape[0]
    p_greater = (np.sum(T_null >= stat_observed) + 1) / (n_perms + 1)
    p_less = (np.sum(T_null <= stat_observed) + 1) / (n_perms + 1)
    p_abs = (np.sum(np.abs(T_null) >= np.abs(stat_observed)) + 1) / (n_perms + 1)
    pvals = pd.Series([p_greater, p_less, p_abs], index=['p_greater', 'p_less', 'p_abs'])

    # Collect outputs.
    Results.t_obs = stat_observed
    Results.p_vals = pvals
    if return_null:
        Results.t_null = T_null
    if return_model:
        Results.model = e_OLfit 

    return Results

def test_community_diffs(feat_array, feat_surrogates, partition, return_null=True, return_model=False):
    """
    Test for a main effect of community on feature.
    
    Parameters
    ----------
    feat_array : array-like
        Feature to test within/between preference for
    feat_surrogates : array_like
        Precomputed surrogates for feat_array
    partition : array_like
        Vector of community assignments

    Returns
    -------
    e_F : float
        Observed F-value for the effect of community
    F_null : array-like
        Null distribution of F-values.
    p_vals : Series
        p-values for the observed effect relative to the null.

    """
    import pandas as pd
    import pingouin as pg
    import numpy as np
    import statsmodels.stats.oneway as smo

    # Create a class to store outputs
    class ResultsStore:
        f_obs = None
        f_null = None
        p_vals = None
        model = None
    Results = ResultsStore()

    # Drop singletons, if they exist
    cids, cts = np.unique(partition, return_counts=True)
    if any(cts == 1):
        part_mask = np.ones_like(partition)
        for st in cids[cts == 1]:
            part_mask[partition == st] = 0
        part_mask = part_mask.astype(bool)
        partition = partition[part_mask]
        feat_array = feat_array[part_mask]
        feat_surrogates = feat_surrogates[:,part_mask]

    # Run ANOVA with the observed data.
    e_res = smo.anova_oneway(feat_array, partition, welch_correction=True)
    e_F = e_res.statistic

    # Run ANOVAs on surrogates.
    F_null = []
    for surr in feat_surrogates:
        null_res = smo.anova_oneway(surr, partition, welch_correction=True)
        null_F = null_res.statistic
        F_null.append(null_F)
    F_null = np.array(F_null)

     ## Calculate the empirical p-value based on permutations
    stat_observed = e_F
    n_perms = feat_surrogates.shape[0]
    p_greater = (np.sum(F_null >= stat_observed) + 1) / (n_perms + 1)
    p_less = (np.sum(F_null <= stat_observed) + 1) / (n_perms + 1)
    p_abs = (np.sum(np.abs(F_null) >= np.abs(stat_observed)) + 1) / (n_perms + 1)
    pvals = pd.Series([p_greater, p_less, p_abs], index=['p_greater', 'p_less', 'p_abs'])

    # Collect outputs.
    Results.f_obs = stat_observed
    Results.p_vals = pvals
    if return_null:
        Results.f_null = F_null
    if return_model:
        Results.model = e_res

    return Results

def within_between_test(feat_array, feat_surrogates, ed_array, affinity_array, partition, return_null=True, return_model=False):
    """
    Test whether pairwise features similarities are greater between vs. within communities.
    
    Parameters
    ----------
    feat_array : array-like
        Feature to test within/between preference for
    feat_surrogates : array_like
        Precomputed surrogates for feat_array
    ed_array : array_like
        Pairwise distances between elements in feat_array
    affinity_array : array_like
        Connectivity matrix
    partition : array_like
        Vector of community assignments

    Returns
    -------
    e_T : float
        Observed t-value for the effect of connection type (within vs between).
    T_null : array-like
        Null distribution of t-values.
    p_vals : Series
        p-values for the observed effect relative to the null.
    e_OLfit: class
        Statsmodels model for the empirical data.

    Notes
    -----
    Setting cov_type='HC2' is equivalent to running a Welch's t-test which does not assume equal variances.
    This is necessary because the number of connections within vs. between differ greatly.
    """
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from itertools import combinations

    # Create a class to store outputs
    class ResultsStore:
        t_obs = None
        t_null = None
        p_vals = None
        model = None
    Results = ResultsStore()

    mask_mat = np.zeros_like(affinity_array)
    cid_array, node_counts = np.unique(partition, return_counts=True)

    # Get within-community edges for all non-singleton communities.
    for cid in cid_array[node_counts > 1]:
        cidx = np.where(partition == cid)
        rows, cols = zip(*list(combinations(cidx[0], 2)))
        mask_mat[rows, cols] = 1
        
    within_mask_triu = mask_mat[np.triu_indices_from(mask_mat, k=1)].astype(bool)

    e_diffs = calc_diffs(feat_array)
    e_OLfit = sm.OLS(e_diffs, np.array([np.ones_like(e_diffs), within_mask_triu.astype(int), ed_array]).T).fit(cov_type='HC2')
    e_T = e_OLfit.tvalues[1]

    T_null = []
    for surr in feat_surrogates:
        n_diffs = calc_diffs(surr)
        n_OLfit = sm.OLS(n_diffs, np.array([np.ones_like(n_diffs), within_mask_triu.astype(int), ed_array]).T).fit(cov_type='HC2')
        T_null.append(n_OLfit.tvalues[1])
    T_null = np.array(T_null)

    ## Calculate the empirical p-value based on permutations
    stat_observed = e_T
    n_perms = feat_surrogates.shape[0]
    p_greater = (np.sum(T_null >= stat_observed) + 1) / (n_perms + 1)
    p_less = (np.sum(T_null <= stat_observed) + 1) / (n_perms + 1)
    p_abs = (np.sum(np.abs(T_null) >= np.abs(stat_observed)) + 1) / (n_perms + 1)
    pvals = pd.Series([p_greater, p_less, p_abs], index=['p_greater', 'p_less', 'p_abs'])

    # Collect outputs.
    Results.t_obs = stat_observed
    Results.p_vals = pvals
    if return_null:
        Results.t_null = T_null
    if return_model:
        Results.model = e_OLfit

    return Results

def flatten(foo):
    # Taken from https://stackoverflow.com/a/5286571
    for x in foo:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in flatten(x):
                yield y
        else:
            yield x

def gen_diffs(feature_array):
    import numpy as np
    diffs = []
    for item in feature_array:
        diffs.append(calc_diffs(item))
    diffs = np.array(diffs)

    return diffs 

def manly_corr(y_data, edges_df, x_var, z_var, dropz=False):
    import sys
    sys.path.append('/home/despo/dlurie/Projects/PyPALM/')
    import pypalm as pm

    edges_df = edges_df.copy()
    y_raw = y_data.y_raw
    y_perms = y_data.y_perms
    edges_df['diffs'] = calc_diffs(y_raw)
    surr_diffs = gen_diffs(y_perms)

    ## If dropz is True, ignore pairs with an affinity of 0
    if dropz:
        zidx = edges_df[x_var] == 0
        edges_df = edges_df[~zidx]
        surr_diffs = surr_diffs[:,~zidx]

    res = pm.manly(edges_df, 'diffs', [x_var], [z_var], stat='pcorr',
                   n_perms=10000, surrogates=surr_diffs, return_null=True)
    return res

## !! DEPRECIATED !! ##
def null_ttest(feat_array, affinity_array, n_perms=10000, dropz=True):
    """
    Use permutations to test the hypothesis that observed pairwise affinity-normalized differences between elements are significantly different than expected by chance.
    
    Parameters
    ----------
    feat_array : array_like
        Array of shape (n_regions,) containing one feature value for each region
    affinity_array : array_like
        Array of shape (n_regions, n_regions) containing standardized (ranging 0 to 1) pairwise affinities (e.g. correlations) between regions.
    n_perms : bool
        Number of null correlations to generate.
        
    Returns
    -------
    ttest : dataframe
        Results of a t-test comparing the observed and null distribution of pairwise normalized differences.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pingouin as pg
    
    # Get the observed normalized differences
    tmp, observed_norm_diffs = calc_diffs(feat_array, affinity_array, permute=False)

    # Generate a distribution of t-values comparing observed to permuted normalized differences.
    tvals = []
    null_store = []
    for i in range(n_perms):
        tmp, null_norm_diffs = calc_diffs(feat_array, affinity_array, permute=True)
        null_store.extend(null_norm_diffs)
        null_ttest = pg.ttest(observed_norm_diffs, null_norm_diffs, correction='auto', tail='two-sided')
        tvals.append(null_ttest['T'].values[0])
    tvals = np.array(tvals)
    null_store = np.array(null_store)
    
    ## Plot null distribution of t-values
    sns.distplot(tvals)
    
    ## If dropz is True, ignore pairs with an affinity of 0
    if dropz:
        null_store = null_store[null_store > 0]
        observed_norm_diffs = observed_norm_diffs[observed_norm_diffs > 0]

    # Run a t-test on the observed vs. null distribution of normalized differences
    ttest = pg.ttest(observed_norm_diffs, null_store, correction='auto', tail='two-sided')
    
    return ttest