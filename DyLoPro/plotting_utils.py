import pandas as pd
import numpy as np

def select_string_column(log, case_agg, col, case_id_key = 'case:concept:name'):
    """
    Extract N columns (for N different attribute values; hotencoding) for the features 
    dataframe for the given string attribute.

    Code of this function inspired by source code of PM4PY. 

    Parameters
    --------------
    log
        Dataframe
    case_agg
        Feature dataframe
    col
        String column
    case_id_key
        Case ID key

    Returns
    --------------
    case_agg
        Feature dataframe (desidered output)
    """
    vals = log.dropna(subset=col)[col].unique()
    for val in vals:
        if val is not None:
            filt_log_cases = log.loc[log[col] == val, case_id_key].unique()
            if type(val)!=bool:
                new_col = col + "_" + val.encode('ascii', errors='ignore').decode('ascii').replace(" ", "")
            elif type(val)==bool:
                if val:
                    new_col = col + "_" + "True"
                else:
                    new_col = col + "_" + "False"
            case_agg[new_col] = case_agg[case_id_key].copy().isin(filt_log_cases)
            case_agg[new_col] = case_agg[new_col].copy().astype("int")
    return case_agg

def select_number_column(log, case_agg, col, numEventFt_transform, 
                         case_id_key = 'case:concept:name') -> pd.DataFrame:
    """
    Extract a column for the features dataframe for the given numeric attribute.

    Code of this function inspired by source code of PM4PY. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log. 
    case_agg
        Feature dataframe
    col
        Numeric column
    numEventFt_transform : {'last', 'first', 'mean', 'median', 'sum', 'prod', 'min', 'max'}
        Determines the way in which these numerical event features are transformed to the 
        case level. By default 'last'. 
    case_id_key
        Case ID key

    Returns
    ----------
    case_agg
        Feature dataframe (desidered output)
    """
    
    log_agg = log.dropna(subset=[col]).groupby(case_id_key, sort=False)[col].agg(numEventFt_transform).reset_index().copy()
    case_agg = case_agg.merge(log_agg, on=[case_id_key], how="left", suffixes=('', '_y'))
    return case_agg

def get_features_df(log: pd.DataFrame, column_list,
                    case_id_key = 'case:concept:name', numEventFt_transform = 'last') -> pd.DataFrame:
    """
    Given a dataframe and a list of columns, performs an automatic feature extraction

    Code of this function inspired by source code of PM4PY. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log. 
    column_list : list of str
        The column names (in 'log') for which the event features need to be elevated 
        to the case level. 
    case_id_key : str, optional
        Column name (in 'log') that contains the case ID. By default 'case:concept:name'.
    numEventFt_transform : {'last', 'first', 'mean', 'median', 'sum', 'prod', 'min', 'max'}
        If any numeric event features contained in 'column_list', 'numEventFt_transform' 
        determines the way in which these numerical event features are transformed to the 
        case level. By default 'last'. 

    Returns
    ----------
    case_agg
        Feature dataframe (desidered output)
    """
    case_agg = pd.DataFrame({case_id_key: list(log[case_id_key].unique())})
    for col in column_list:
        data_tp = str(log[col].dtype)
        if "obj" in data_tp or "str" in data_tp or "bool" in data_tp or "categ" in data_tp:
            case_agg = select_string_column(log, case_agg, col, case_id_key=case_id_key)
        elif "float" in data_tp or "int" in data_tp:
            case_agg = select_number_column(log, case_agg, col, numEventFt_transform, case_id_key=case_id_key)
    return case_agg

def _event_fts_to_tracelvl(log, event_features, numEventFt_transform = 'last'):
    """Transforms categorical and numerical event features to the trace level.

    Parameters
    ----------
    log : pandas.DataFrame
        Current event log. 
    event_features : list of str
        Column names of the event features that need to be elevated to the trace level. 
    numEventFt_transform : {'last', 'first', 'mean', 'median', 'sum', 'prod', 'min', 'max'}
        If any numeric event features contained in 'event_features', 'numEventFt_transform' 
        determines the way in which these numerical event features are transformed to the 
        case level. By default 'last'. 

    Returns
    -------
    _type_
        _description_
    """
    '''Transforms categorical and numeric event features to the trace level.
        args:
            - log               : pd.DataFrame
            - event_features    : list of strings, containing the column names of the event features to be preprocessed. 

        returns:
            - log: enhanced pd.DataFrame in which the event features are transformed to the trace level:
                    - numeric event features: last recorded (non-null) value for each trace taken; name= [original name]+'_trace'
                    - categorical event features: binary column (0 - 1) added for each level; names = [original name]+'_'+[level name]
    '''
    feature_table = get_features_df(log, event_features, numEventFt_transform = numEventFt_transform)
    log = log.merge(feature_table, on='case:concept:name', how='left', suffixes=("","_trace"))
    return log


def determine_time_col(frequency, case_assignment):
    """Determine the time period column ('time_col') that indicates 
    to which time period each case should be assigned too. 

    Parameters
    ----------
    frequency : {'minutely', '5-minutely', '10-minutely', 
                'half-hourly', 'hourly' '2-hourly', '12-hourly', 'daily', 
                'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Determines the intervals of time periods to which cases are 
        assigned.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Given the time intervals determined by 'frequency', 
        'case_assigment' determines the the condition upon which
        each case is assigned to a certain time period. 

    Returns
    -------
    time_col : str
        Time period column that contains for each case the time period
        it should be assigned to. 
    """
    frequencies = ['minutely', '5-minutely', '10-minutely', 'half-hourly', 
                   'hourly', '2-hourly', '12-hourly', 'daily', 'weekly', 
                   '2-weekly', 'monthly', 'quarterly', 'half-yearly']
    
    periodic_timestamps = ['1min_timestamp', '5min_timestamp', '10min_timestamp', 
                           '30min_timestamp', 'hourly_timestamp', '2hour_timestamp', 
                           '12hour_timestamp', 'daily_timestamp', 'weekly_timestamp', 
                           'two_weekly_timestamp', 'monthly_timestamp', 
                           'quarterly_timestamp', 'half_yearly_timestamp']

    freq_idx = frequencies.index(frequency)
    time_col = periodic_timestamps[freq_idx]


    case_assignments = ['first_event', 'last_event', 'max_events']
    assignment_suffixes = ['_firstev', '_lastev', '_maxev']

    cassi_idx = case_assignments.index(case_assignment)
    time_col = time_col + assignment_suffixes[cassi_idx]

    return time_col 

def determine_tt_col(time_unit):
    '''
        args: 
            - time_unit: string = 'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days' or 'weeks'; time unit for Throughput Time 

        returns:
            - tt_col: string ; name of the throughput time (tt) column with correct time unit. 
    '''
    tt_cols = ['tt_microseconds', 'tt_milliseconds', 'tt_seconds', 'tt_minutes', 'tt_hours', 'tt_days', 'tt_weeks']
    time_unit_list = ['microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks']
    time_unit_idx = time_unit_list.index(time_unit)
    tt_col = tt_cols[time_unit_idx]
    
    return tt_col

def get_outcome_percentage(filtered_log, outcome, time_col):
    ''' Computes the periodic percentage of cases with outcome == 1 for a filtered log
        args:
            - filtered_log: pd.DataFrame; case log (i.e. log with 1 row per case) that is filtered based on a condition.
            - outcome:      string of the outcome column. (Has to be a binary outcome column with 0 and 1's)
            - time_col:     string of the periodic timestamp column.

        returns:
            - periodic % of (the filtered) cases with outcome == 1 
    '''
    # Periodic number of cases in the filtered log. (I.e. periodic number of cases that satisfy the condition on which you have filtered.)
    per_counts = filtered_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    per_counts.columns = ['total']
    # Filtering the filtered cases on outcome == 1
    filtered_case = filtered_log[filtered_log[outcome]==1]
    # Periodic number of those filtered cases with outcome == 0 and outcome == 1
    per_true = filtered_case.pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
    if len(per_true) == 0:
    # if 1 not in dfr_true.columns:
        per_counts['num_true'] = [0 for i in range(len(per_counts))]
    else:
        per_true.columns = ['num_true']
        per_counts = per_counts.merge(per_true, left_index = True, right_index = True, how='left')

    per_counts['prc_true'] = per_counts['num_true'] / per_counts['total']

    return per_counts[['prc_true']]

def get_dfr_time(log, case_log, dfr_list, time_col, numeric_agg):
    ''' For each case: computes the performance (time between) for each occurance of the given dfr (if any).
        args: 
            - log: pd.DataFrame 
            - dfr_list: list of tuples; [('activity_string_1', 'activity_string_2'), ...]

        returns: 
            - period_dfr_perf:  pd.DataFrame; aggregated periodic performances (time between) for each of the given dfr's in dfr_list
            - perf_units_cols:  list of strings; indicating the automatically determined time_unit for each of the columns / dfr's. The possibilities
                                are 'days', 'hours', 'minutes', 'seconds', 'milliseconds' and 'microseconds'. 
    '''

    time_products = [1, 24, 1440, 86400, 86400000, 86400000000]
    perf_units = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds']

    
    log_loc = log.copy()
    log_loc['next:concept:name'] = log_loc.groupby(['case:concept:name'])['concept:name'].shift(-1)
    log_loc['dfr_start'] = list(zip(log_loc['concept:name'], log_loc['next:concept:name']))
    log_loc['next_stamp']= log_loc.groupby(['case:concept:name'])['time:timestamp'].shift(-1)
    log_loc['time_till_next'] = (log_loc['next_stamp'] - log_loc['time:timestamp']) / pd.Timedelta(days=1) # In days (int)
    log_filtered = log_loc[log_loc['dfr_start'].isin(dfr_list)][['case:concept:name', 'dfr_start', 'time_till_next']].copy()
    log_filtered = log_filtered.merge(case_log[['case:concept:name', time_col]], on= 'case:concept:name', how= 'left')
    period_dfr_perf = log_filtered.pivot_table(values = 'time_till_next', index = time_col, columns = 'dfr_start', aggfunc = numeric_agg, fill_value = 0) # in days
    # Reordering corresponding to order in dfr_list 
    period_dfr_perf = period_dfr_perf[dfr_list]
    # Renaming:
    col_strings = [str(col)+'_perf' for col in list(period_dfr_perf.columns)]
    period_dfr_perf.columns = col_strings
    col_means = np.mean(period_dfr_perf, axis = 0).to_numpy().astype(np.float64)

    time_products = np.array(time_products, dtype= np.float64)
    # Broadcasting to the same shape for multiplication: 
    num_units = 6
    num_cols = col_means.shape[0]
    col_means_bc = np.broadcast_to(col_means[:, None], (num_cols, num_units))
    time_products_bc = np.broadcast_to(time_products[None, :], (num_cols, num_units))
    # Multiplication and determining the right time unit by taking the result closest to 1: 
    product_matrix = np.multiply(col_means_bc, time_products_bc) # (num_cols, num_units)
    # For each of the performance columns, select the one closest to 1. 
    time_indices = np.argmin(np.abs(product_matrix - 1), axis=1) # (num_cols, ) 

    time_prods_res = np.take(time_products, indices= time_indices)
    time_prods_resbc = np.broadcast_to(time_prods_res[None, :], (period_dfr_perf.shape[0], num_cols))
    period_dfr_perf = np.multiply(period_dfr_perf[col_strings], time_prods_resbc)
    perf_units_cols = [perf_units[idx] for idx in time_indices] # (num_cols, )

    return period_dfr_perf, perf_units_cols 

def get_maxrange(agg_df):
    ''' Computes for each of the given arrays / pd Series the maximum y-range, by neglecting extreme outliers (values > q3 + 3*iqr). 
        args:
            - agg_df: (num periods, num series) - shaped dataframe, containing the series for which outliers should be accounted for before plotting. 
        
        returns: 
            - max_values:   (num series, ) - shaped np.ndarray, containing the max y-values < q3 + 3*iqr for each of the columns / series in agg_df. 
    '''

    q1 = np.quantile(agg_df, 0.25, axis = 0) # shape (num series, )
    q3 = np.quantile(agg_df, 0.75, axis = 0) # shape (num series, )
    iqr = q3 - q1 
    upper_bound = q3 + (3*iqr)
    # Replacing the extreme outliers in each column by NaN:
    agg_df_MinusFliers = np.where(agg_df>upper_bound[None,:], np.nan, agg_df) # shape (num periods, num series)

    return np.nanmax(agg_df_MinusFliers, axis=0)  # max_values: shape (num series, )


def get_variant_case(log):
    ''' Takes the whole log as input, and returns a pd.DataFrame with the case id in column 0, and the variant (as a tuple) in column 1
        
        args:
            - log: pd.DataFrame

        returns:
            - case_variant: pd.DataFrame with 1 row for each case and 2 columns: 'case:concept:name' and 'variant'. 
    '''
    log_loc = log[['case:concept:name', 'concept:name']].copy()
    case_variant = log_loc.groupby(['case:concept:name'], as_index = False, sort = False).agg({'concept:name': ','.join})
    case_variant.columns = ['case:concept:name', 'variant']
    case_variant['variant'] = case_variant['variant'].apply(lambda x: tuple(x.split(',')))
    return case_variant

def get_tt_ratios(log, num_fts_list, time_col, numeric_agg):
    ''' Computes the periodic ratio of throughput time over numerical feature and automatically determines the most appropriate time unit for that ratio 
        for each of the numerical features in num_fts_list simultaneously. 
        args: 
            - log               :   pd.DataFrame 
            - num_fts_list      :   list of strings; containing the column names of the numerical features. 

        returns: 
            - period_ttr        :   pd.DataFrame; aggregated periodic ratios for each of the given numerical features in num_fts_list
            - perf_units_cols   :   list of strings; indicating the automatically determined time_unit for each of the columns / given numerical features. The possibilities
                                    are 'days', 'hours', 'minutes', 'seconds', 'milliseconds' and 'microseconds'. 
    '''
    log_loc = log.copy()
    case_log_cop = log_loc.drop_duplicates(subset = 'case:concept:name').copy()
    time_products = [1, 24, 1440, 86400, 86400000, 86400000000]
    ratio_units = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds']
    num_units = 6

    fts_list_ext = num_fts_list.copy()
    fts_list_ext.extend(['case:concept:name', time_col, 'tt_days']) 
    case_log_loc = case_log_cop[fts_list_ext].copy()
    cols = list(case_log_loc.columns)
    case_log_loc[num_fts_list] = np.multiply((1 / case_log_loc[num_fts_list]), np.asarray(case_log_loc[['tt_days']]))
    # Replacing the np.inf values by np.nan values
    case_log_loc = pd.DataFrame(np.where(case_log_loc == np.inf, np.nan, case_log_loc ))
    case_log_loc.columns = cols
    period_ttr = case_log_loc.pivot_table(values = num_fts_list, index = time_col, aggfunc = numeric_agg, fill_value = 0)
    # Reordering
    period_ttr = period_ttr[num_fts_list]
    col_strings = [str(col)+'_tt_ratio' for col in list(period_ttr.columns)]
    period_ttr.columns = col_strings
    # Getting the column means 
    col_means = np.mean(period_ttr, axis = 0).to_numpy().astype(np.float64)
    num_cols = col_means.shape[0]
    
    time_products = np.array(time_products, dtype= np.float64)
    
    col_means_bc = np.broadcast_to(col_means[:, None], (num_cols, num_units))
    time_products_bc = np.broadcast_to(time_products[None, :], (num_cols, num_units))
    # Multiplication and determining the right time unit by taking the result closest to 1: 
    product_matrix = np.multiply(col_means_bc, time_products_bc) # (num_cols, num_units)
    # For each of the performance columns, select the one closest to 1. 
    time_indices = np.argmin(np.abs(product_matrix - 1), axis=1) # (num_cols, ) 
    time_prods_res = np.take(time_products, indices= time_indices)
    time_prods_resbc = np.broadcast_to(time_prods_res[None, :], (period_ttr.shape[0], num_cols))
    period_ttr = np.multiply(period_ttr[col_strings], time_prods_resbc)
    ratio_units_cols = [ratio_units[idx] for idx in time_indices] # (num_cols, )

    return period_ttr, ratio_units_cols 

def get_sorted_DFRs(log):
    ''' Retrieves all the Directly-Follows Relations (DFRs) in the log, sorts them from highest to lowest frequency, 
        and returns them as a list of tupples. E.g. [('act_1_dfr_1', 'act_2_dfr_1'), ('act_1_dfr_2', 'act_2_dfr_2'), ...]

        args:
            - log       :   pd.DataFrame
        
        returns:
            - list_dfr  :   sorted list of all the DFRs, as specified here above. 
    '''
    locallog = log[['case:concept:name', 'concept:name']].copy()
    locallog['next:concept:name'] = locallog.groupby('case:concept:name')['concept:name'].shift(-1)
    locallog = locallog.dropna(subset='next:concept:name').copy()
    locallog['dfr_start'] = list(zip(locallog['concept:name'], locallog['next:concept:name']))
    dfr_df = pd.DataFrame(locallog['dfr_start'].value_counts()).reset_index()
    dfr_df.columns = ['DFR', 'DFR count']
    list_dfr = list(locallog['dfr_start'].value_counts().index)

    return list_dfr, dfr_df 

def get_filtered_dfr_df(dfr_df, max_k = None, directly_follows_relations = None, counts = False):
    """Return a `pandas.DataFrame` containing the DFR numbers together 
    with a tuple containing the two corresponding activity labels 
    of each DFR. Called by the `get_DFR_df()` method of the 
    `DynamicLogPlots` class. 

    Parameters
    ----------
    dfr_df: pandas.DataFrame
        Dataframe containing a row for each DFR, with each row 
        having two columns: 'DFR' and 'DFR count', containing 
        the DFR tuple and amount of occurrences over the whole
        log respectively. The rows are arranged in descending 
        order of 'DFR count'. `dfr_df` can be obtained by means
        of the `get_sorted_DFRs(log)` function. 
    max_k : int, optional
        If specified, a dataframe containing the DFR numbers and 
        their corresponding activity pairs for the `max_k` most 
        frequently occurring DFRs is returned. By default `None`.
    directly_follows_relations : list of tuple, optional
        List of tuples containing the specified DFRs. Each DFR 
        needs to be specified as a tuple that contains 2 strings, 
        referring to the 2 activities in the DFR, e.g. 
        ('activity_a', 'activity_b'). If specified and `max_k=None`, 
        a dataframe containing the DFR numbers and corresponding 
        activity pairs for each of the specified DFRs is returned. 
        If `max_k!=None`, the DFRs specified here are ignored. 
        By default `None`.
    counts : bool, optional
        If `counts=True`, the 'DFR count' column that contains the 
        number of occurrences of each DFR over the whole event log 
        is included in the returned dataframe. 
    Returns
    -------
    filtered_df_df : pandas.DataFrame
        Dataframe containing 'DFR number' index and the 'DFR' column 
        containing for each requested DFR the encoded DFR index and 
        the DFR tuple containing the corresponding activity pair 
        respectively. If `counts=True`, the 'DFR count' column that 
        contains the number of occurrences of each DFR over the 
        whole event log is included too. 
    """
    if max_k:
        filtered_dfr_df = dfr_df.iloc[:max_k, :].copy()
    else: 
        filtered_dfr_df = dfr_df[dfr_df['DFR'].isin(directly_follows_relations)].copy().reset_index(drop = True )
    # Adding the encoded 'DFR number' column
    filtered_dfr_df['DFR number'] = [i for i in range(1, len(filtered_dfr_df)+1)]
    filtered_dfr_df.set_index('DFR number', inplace= True)
    
    if counts: 
        return filtered_dfr_df
    
    return filtered_dfr_df[['DFR']]

def get_filtered_var_df(variant_df, max_k, variants, counts):
    """Get a `pandas.DataFrame` containing the variant numbers together 
    with a tuple containing the activity label strings
    of each variant. 

    Parameters
    ----------
    var_df : pandas.DataFrame 
        Dataframe containing a row for each variant, with each row 
        having two columns: 'variant' and 'variant count', containing 
        the variant tuple and amount of occurrences over the whole
        log respectively. The rows are arranged in descending 
        order of 'variant count'.
    max_k : int, optional
        If specified, a dataframe containing the encoded variant 
        numbers and corresponding tuple of N strings (with N the 
        number of activities of a variant) is returned for 
        each of the `max_k` most frequently occurring variants. 
        By default `None`.
    variants : list of tuple
        The variants for which the requested time series will be 
        plotted. Each variant needs to be specified as a tuple that 
        contains N strings, referring to the N activities that 
        constitute that variant. If specified and `max_k=None`, 
        a dataframe containing the encoded variant numbers and 
        corresponding tuple of N strings for each of the specified 
        DFRs is returned. If `max_k!=None`, the `variants` parameter 
        is ignored. By default `None`.
    counts : bool, optional
        If `counts=True`, the 'variant count' column that contains the 
        number of occurrences of each variant over the whole event log 
        is included in the returned dataframe. 

    Returns
    -------
    filtered_var_df : pandas.DataFrame
        Dataframe containing 'variant number' index and the 'variant' 
        column containing for each requested variant the encoded variant  
        index and the variant tuple containing the corresponding activity 
        pair respectively. If `counts=True`, the 'variant count' column 
        that contains the number of occurrences of each variant over the 
        whole event log is included too. 
    """
    if max_k:
        filtered_var_df = variant_df.iloc[:max_k, :].copy()
    else: 
        filtered_var_df = variant_df[variant_df['variant'].isin(variants)].copy().reset_index(drop = True )
    # Adding the encoded 'variant number' column
    filtered_var_df['variant number'] = [i for i in range(1, len(filtered_var_df)+1)]
    filtered_var_df.set_index('variant number', inplace= True)
    
    if counts: 
        return filtered_var_df
    
    return filtered_var_df[['variant']]
    

def get_uniq_varcounts(log, time_col):
    ''' Computes the number of distinct variants in each time bucket, as well as the number of distinct previously 
        unseen / distinct new variants in each time bucket. 

        args:
            - log               :   pd.DataFrame, already enhanced with a column that contains the variant of that case. 
            - time_col          :   string; indicating the column in which the 'periodic group' of that case is documented.
        
        returns:
            - period_varcounts  :   pd.DataFrame containing:
                                    -   the sorted time periods / buckets as the indices
                                    -   the number of distinct variants in each time bucket / period as the 'num_distinct_vars' column
                                    -   the number of distinct previously unseen / dinstinct new variants in each time bucket as the
                                        'num_NEW_distinct_vars' column 
    '''
    # First the amount of distinct variants (not per se unseen) in each period:
    local_log = log[['case:concept:name', time_col, 'variant']].copy()
    local_log = local_log.drop_duplicates(subset= 'case:concept:name')
    period_numDistinctvars = local_log.pivot_table(values ='variant', index = time_col, aggfunc = pd.Series.nunique, fill_value = 0)

    local_log = local_log.sort_values(time_col)
    local_log = local_log.drop_duplicates(subset = 'variant')
    # We now have a dateframe in which each variant only occurs once, sorted by the periodic timestamp col. Hence,
    # taking the number of unique distinct variants in each time period (time_col) gives us the newly introduced variants in that time period. 
    period_numUnseenVars= local_log.pivot_table(values ='variant', index = time_col, aggfunc = pd.Series.nunique, fill_value = 0)

    # merging both
    period_numDistinctvars.columns = ['num_distinct_vars']
    period_numUnseenVars.columns = ['num_NEW_distinct_vars']
    period_varcounts = period_numDistinctvars.merge(period_numUnseenVars, left_index = True, right_index = True, how = 'left').fillna(0)

    return period_varcounts

def get_ordered_variantMap(case_log, time_col):
    ''' Computes a variant mapping that maps the tuple representation of a variant to a unique string. The mapping is ordered. 

        args:
            - case_log          :   pd.DataFrame, 1 row per case, already enhanced with a column that contains the variant of that case. 
            - time_col          :   string; indicating the column in which the 'periodic group' of that case is documented.
        
        returns:
            - case_log               :   pd.DataFrame containing an additional column: variant_str'
            - periodic_newvars  :   pd.DataFrame containing 2 columns: 1st column containing the unique periodic timestamps (determined by time_col),
                                    and the 2nd column containing the tuple of (mapped) variant strings containing the previously unseen variants that
                                    occurred for the first time in that time period. 
    '''
    local_log = case_log[['case:concept:name', 'time:timestamp', time_col, 'variant']].copy()
    local_log = local_log.drop_duplicates(subset = 'case:concept:name')
    local_log = local_log.sort_values([time_col, 'time:timestamp'])
    local_log = local_log.drop_duplicates(subset = 'variant')
    local_log.loc[:,'variant_str'] = ['Variant_{}'.format(i) for i in range(len(local_log))]
    variant_map = local_log[['variant', 'variant_str']]
    case_log = case_log.merge(variant_map, on = 'variant', how = 'left')

    # Now we can also compute the periodic newvars (a dataframe with the time_col and for each period the tuple of variant strings that were introduced for the first time 
    # in that period).
    periodic_newvars = local_log.groupby([time_col], as_index = False, sort = False).agg({'variant_str': ','.join})
    periodic_newvars.columns = [time_col, 'new_variants']
    periodic_newvars.loc[:, 'new_variants'] = periodic_newvars.loc[:, 'new_variants'].apply(lambda x: tuple(x.split(',')))
    
    return case_log, periodic_newvars

def get_newVar_cases(log, time_col):
    ''' Determines which cases belong to variants that were introduced for the first time during its corresponding periodic timestamp (determined by time_col). 

        args:
            - log               :   pd.DataFrame, already enhanced with a column that contains the variant of that case. 
            - time_col          :   string; indicating the column in which the 'periodic group' of that case is documented.
        
        returns:
            - log               :   pd.DataFrame containing an additional column: variant_str'
            - periodic_newvars  :   pd.DataFrame containing 2 columns: 1st column containing the unique periodic timestamps (determined by time_col),
                                    and the 2nd column containing the tuple of (mapped) variant strings containing the previously unseen variants that
                                    occurred for the first time in that time period. 
    '''
    loc_log = log.drop_duplicates(subset = 'case:concept:name').copy()
    case_id_log = loc_log[['case:concept:name']].copy()
    enhanced_log, periodic_newvars = get_ordered_variantMap(case_log = loc_log, time_col = time_col)
    time_col_list = periodic_newvars[time_col].unique()
    newVar_cases = []
    for period in time_col_list:
        new_vars = periodic_newvars.loc[periodic_newvars[time_col]==period,'new_variants'].item()
        log_sliced = enhanced_log[enhanced_log[time_col]==period].copy()
        newVar_cases_period = list(log_sliced.loc[log_sliced['variant_str'].isin(new_vars), 'case:concept:name'].unique())
        newVar_cases += newVar_cases_period
    # So now we have the list of all case_id's for that correspond to a variant that is newly introduced in its respective time period
    case_id_log.loc[:, "NewOrOld"] = np.where(case_id_log['case:concept:name'].isin(newVar_cases), "New", "Old")

    return case_id_log