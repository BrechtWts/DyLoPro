import pandas as pd
import numpy as np

def select_string_column(log, 
                         case_agg, 
                         col, 
                         case_id_key = 'case:concept:name'):
    """
    Extract N columns (for N different attribute values; hotencoding) for 
    the features dataframe for the given string attribute.

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
    Extract a column for the features dataframe for the given numeric 
    attribute.

    Code of this function inspired by source code of PM4PY. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log.

    case_agg
        Feature dataframe.

    col
        Numeric column.

    numEventFt_transform : {'last', 'first', 'mean', 'median', 'sum', 'prod', 
                            'min', 'max'}
        Determines the way in which these numerical event features are 
        transformed to the case level. By default 'last'.
        
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
    Given a dataframe and a list of columns, performs an automatic feature 
    extraction.

    Code of this function inspired by source code of PM4PY. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log.

    column_list : list of str
        The column names (in 'log') for which the event features need to be 
        elevated to the case level.

    case_id_key : str, optional
        Column name (in 'log') that contains the case ID. By default 
        'case:concept:name'.

    numEventFt_transform : {'last', 'first', 'mean', 'median', 'sum', 'prod', 
                            'min', 'max'}
        If any numeric event features contained in ``column_list``, 
        ``numEventFt_transform`` determines the way in which these numerical 
        event features are transformed to the case level. By default 'last'.

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

def _event_fts_to_tracelvl(log, 
                           event_features, 
                           numEventFt_transform = 'last'):
    """Transforms categorical and numerical event features to the trace level.

    Parameters
    ----------
    log : pandas.DataFrame
        Current event log.

    event_features : list of str
        Column names of the event features that need to be elevated to the 
        trace level.

    numEventFt_transform : {'last', 'first', 'mean', 'median', 'sum', 'prod', 
                            'min', 'max'}
        If any numeric event features contained in ``event_features``, 
        ``numEventFt_transform`` determines the way in which these 
        numerical event features are transformed to the 
        case level. By default 'last'. 

    Returns
    -------
    log : pd.DataFrame
        Enhanced event log in which the event features are transformed to the 
        trace level. 

        For:

        * numeric event features: For each trace, all **non-null** 
          occurences of that feature are aggregated into a single case 
          measure based on the ``numEventFt_transform`` argument. If a 
          case only contains null values for that feature, it is 
          assigned an NaN value. 

        * categorical event features: A binary column (0-1) is added for each 
          level of that feature. For each case, a value of 1 is assigned if 
          that level occurs at least in one of its events, and 0 otherwise. 
          It is important to note that, for categorical event features, this 
          function is only called with a reduced event log, in which only the 
          levels of interest for a certain categorical event feature are 
          retained, while all other levels are discarded. In that way, 
          high-level categoricals do not cause memory overload and / or 
          computational bottlenecks. As such, for categorical event features, 
          this function is only called for one feature at a time. 

    """
    feature_table = get_features_df(log, event_features, numEventFt_transform = numEventFt_transform)
    log = log.merge(feature_table, on='case:concept:name', how='left', suffixes=("","_trace"))
    return log


def determine_time_col(frequency, case_assignment):
    """Determine the time period column ('time_col') that indicates 
    to which time period each case should be assigned too. 

    Parameters
    ----------

    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
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
    """Determine the appropriate time unit column for the throughput time 
    based on the ``time_unit`` argument specified by the user. 

    Parameters
    ----------

    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 
                'hours', 'days', 'weeks'}
        Time unit for the throughput time specified by the user. 

    Returns
    -------
    tt_col : str
        Column name throughput time with correct unit in the preprocessed 
        event log. 
    """
    tt_cols = ['tt_microseconds', 'tt_milliseconds', 'tt_seconds', 'tt_minutes', 'tt_hours', 'tt_days', 'tt_weeks']
    time_unit_list = ['microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks']
    time_unit_idx = time_unit_list.index(time_unit)
    tt_col = tt_cols[time_unit_idx]
    
    return tt_col

def get_outcome_percentage(filtered_log, outcome, time_col):
    """Compute for each time bucket the fraction of cases with 
    ``outcome=1`` for the cases in ``filtered_log``. 

    Parameters
    ----------
    filtered_log : pd.DataFrame
        FIltered event log. The case filtering is performed by the plotting 
        functions calling this util function, and depends on the purpose and 
        arguments of these plotting functions. 
    outcome : str
        Name of the outcome column in ``filtered_log``.
    time_col : str
        The time column of interest in the event log. Determined in the 
        plotting functions calling this util function. 

    Returns
    -------
    pd.DataFrame
        Periodic fraction of cases with ``outcome=1``. 
    """
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
    """Compute, for each case, the DFR performance (i.e. time elapsed 
    between the occurrence of the first and second activity of a given 
    Directly-Follows Relationship (DFR)) for each occurrence of each 
    DFR in ``dfr_list``. Then, the cases are grouped into discrete 
    time buckets, producing for each DFR a set of DFR performances. 
    Finally, for each of the DFRs in ``dfr_list``, an aggregate 
    DFR perforamnce is computed for each time bucket, with the 
    aggregation function being determined by the ``numeric_agg`` 
    parameter. 

    Parameters
    ----------
    log : pd.DataFrame
        The (preprocessed) event log. 
    case_log : pd.DataFrame
        Dataframe containing only one row for each case. 
    dfr_list : list of tuple 
        The Directly-Follows Relations for which the periodically aggregated 
        DFR performances are computed. Should be formatted as follows: 
        [('activity_string_1', 'activity_string_2'), ...].
        _description_
    time_col : str
        The time column of interest in the event log. Determined in the 
        plotting functions calling this util function. 
    numeric_agg : str
        How the DFRs assigned to each time bucket should be aggregated 
        towards one aggregated measure. 

    Returns
    -------
    period_dfr_perf : pd.DataFrame 
        Dataframe with on the rows the chronologically ordered time buckets, 
        and with each column being the aggregated DFR performances for one of 
        of the DFRs listed in ``dfr_list``.
    perf_units_cols : list of string
        The most suitable time units are, for each requested DFR separately, 
        determined automatically. This list contains these time units in the 
        correct order, such that the appropriate time units can be displayed 
        in the visualizations. 
    """

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
    """Compute for each of the given arrays / pd.Series the maximum value 
    that should not be considered as an extreme outlier (i.e. < q3 + 3 x iqr).

    Parameters
    ----------
    agg_df : pd.DatFrame
        Shape (num periods, num series), with num periods being the amount of 
        time buckets, and num series being the amount of time series for 
        which outliers should be accounted for. 

    Returns
    -------
    max_values : np.ndarray
        The 'num series' maximum values not to be considered as extreme 
        outliers.
    """

    q1 = np.quantile(agg_df, 0.25, axis = 0) # shape (num series, )
    q3 = np.quantile(agg_df, 0.75, axis = 0) # shape (num series, )
    iqr = q3 - q1 
    upper_bound = q3 + (3*iqr)
    # Replacing the extreme outliers in each column by NaN:
    agg_df_MinusFliers = np.where(agg_df>upper_bound[None,:], np.nan, agg_df) # shape (num periods, num series)

    return np.nanmax(agg_df_MinusFliers, axis=0)  # max_values: shape (num series, )


def get_variant_case(log):
    """Return a pd.Dataframe with a row row for each case, containing the 
    unique case ID and the corresponding variant (as a tuple) in its columns.

    Parameters
    ----------
    log : pd.DataFrame
        The event log.

    Returns
    -------
    case_variant : pd.DataFrame
        Dataframe with one row for each case and two columns, 
        'case:concept:name' and 'variant'.
    """
    log_loc = log[['case:concept:name', 'concept:name']].copy()
    case_variant = log_loc.groupby(['case:concept:name'], as_index = False, sort = False).agg({'concept:name': ','.join})
    case_variant.columns = ['case:concept:name', 'variant']
    case_variant['variant'] = case_variant['variant'].apply(lambda x: tuple(x.split(',')))
    return case_variant

def get_tt_ratios(log, num_fts_list, time_col, numeric_agg):
    """Compute the periodic ratio of throughput time over numerical feature 
    and automatically determine the most appropriate time unit for that ratio 
    for each of the numerical features in ``num_fts_list`` simultaneously. 

    Parameters
    ----------
    log : pd.DataFrame
        The event log.
    num_fts_list : list of str
        Column names of numeric features of interest.
    time_col : str
        The time column of interest in the event log. Determined in the 
        plotting functions calling this util function. 
    numeric_agg : str
        How the ratios assigned to each time bucket should be aggregated 
        towards one aggregated measure. 

    Returns
    -------
    period_ttr : pd.DataFrame
        Periodically aggregated TT ratios for each of the numeric features 
        given in `num_fts_list`.
    perf_units_cols : list of str
        Indicating the automatically determined time unit for each of the 
        requested numeric features.
    """
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
    """Retrieve all the Directly-Follows Relations (DFRs) in the log, sort
    them from highest to lowest frequency, and return them as a list of 
    tuples. E.g. [('act_1_dfr_1', 'act_2_dfr_1'), ('act_1_dfr_2', 'act_2_dfr_2'), 
    ...].

    Parameters
    ----------
    log : pd.DataFrame
        The event log.

    Returns
    -------
    list_dfr : str of tuple
        Sorted list of DFRs (in descending order of frequency).
    """
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
    """Compute the number of distinct variants in each consecutive time 
    bucket, as well as the number of distinct previously unseen variants 
    in each time bucket.

    Parameters
    ----------
    log : pd.DataFrame
        Event log that is already enhanced, prior to this function being 
        called, with an additional 'variant' column containing the variant 
        of each case. 
    time_col : str
        The time column of interest in the event log. Determined in the 
        plotting functions calling this util function. 

    Returns
    -------
    periodic_varcounts : pd.DataFrame
        Containing the chronologically ordered time buckets as the indices, 
        and two columns, containing for each time bucket the number of 
        distinct variants, and the number of distinct variants not seen in 
        any of the preceding periods, respectively. 
    """

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
    """Computes a variant mapping that maps the tuple representation of a 
    variant to a unique string. The mapping is ordered. Only used by the 
    ``get_neVar_cases()`` util function in this same module.

    Parameters
    ----------
    case_log : pd.DataFrame
        Event log containing only one row for each case. Already enhanced with 
        a column that contains the variant of that case.
    time_col : str
        Column name specifying the relevant time bucket each case should be 
        assigned to. 

    Returns
    -------
    case_log : pd.DataFrame
        case_log containing an additional column 'variant_str'.
    periodic_newvars : pd.DataFrame 
        Contains two columns, the chronologically ordered unique periodic 
        timestamps / time buckets, and a tuple of mapped variant strings in 
        the second column containing the variants that were first observed 
        in the corresponding time bucket. 
    """
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
    """Determines for each consecutive time period / time bucket the cases 
    pertaining to variants that were first observed in that time period. 
    
    Only called by the ``distinct_variants_AdvancedEvol()`` plotting method. 

    Parameters
    ----------
    log : pd.DataFrame 
        Event log that is already enhanced, prior to this function being 
        called, with an additional 'variant' column containing the variant 
        of each case. 
    time_col : str
        The time column of interest in the event log. Determined in the 
        plotting functions calling this util function. 

    Returns
    -------
    log : pd.DataFrame
        Event log containing additional column 'variant_str'.
    periodic_newvars : pd.DataFrame 
        Contains two columns, the chronologically ordered unique periodic 
        timestamps / time buckets, and a tuple of mapped variant strings in 
        the second column containing the variants that were first observed 
        in the corresponding time bucket. 
    """
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