import DyLoPro.check_args as ca
import pandas as pd
import numpy as np

# Some checks are identical for all plotting functions, these are grouped in the _verify_common() method:
def _verify_common(time_unit, frequency, case_assignment, numeric_agg, xtr_outlier_rem):
    """Verify validity of the hyperparameters common to all plotting methods. 

    Parameters
    ----------
    time_unit : str
        Time unit in which the throughput time of cases is specified. 
        The options are {'microseconds', 'milliseconds', 'seconds', 'minutes', 
        'hours', 'days', 'weeks'}. 
    frequency : str
        Frequency by which the observations are grouped together. The options are
        {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
        '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : str
        Condition upon which a case is assigned to a certain period. 
        Options are {'first_event', 'last_event', 'max_events'}.
    numeric_agg : str
        _description_
    xtr_outlier_rem : bool
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    ''' Verification procedure of the arguments passed to this function is identical for all plotting methods.
        Therefore, these procedures are grouped as one in this function. 

        returns: 
            - error     : bool; True if an argument is specified incorrectly, False otherwise. 
            - err_msge  : string; contains the error-specific error message if any.
    '''
    error, err_msge = False, " "

    # verify time_unit:
    if time_unit != 'days':
        error, err_msge = ca._verify_time_unit(time_unit)
        if error:
            return error, err_msge
    
    # verify frequency 
    if frequency != 'weekly':
        error, err_msge = ca._verify_frequency(frequency)
        if error:
            return error, err_msge
    
    # verify case_assignment
    if case_assignment != 'first_event':
        error, err_msge = ca._verify_case_assignment(case_assignment)
        if error:
            return error, err_msge
    
    # verify numeric_agg
    error, err_msge = ca._verify_numeric_agg(numeric_agg)
    if error:
        return error, err_msge
    
    # verify xtr_outlier_rem:
    error, err_msge = ca._verify_xtr_outlier_rem(xtr_outlier_rem)
    if error:
        return error, err_msge
    
    return error, err_msge



def _verify_topKdfr(outcome, time_unit, frequency, case_assignment, plt_type, numeric_agg, max_k, xtr_outlier_rem):
    '''
    Checks whether all arguments are correctly specified for the topK_dfr_evol() method.

    '''

    if plt_type != 'univariate':
        error, err_msge = ca._verify_typedfr(plt_type)
        if error:
            raise ValueError(err_msge)

    # verify outcome (if plt_type = 'type_outcome):
    if plt_type == 'type_outcome':
        error, err_msge = ca._verify_outcomeCol(outcome)
        if error:
            raise ValueError(err_msge)

    # common verifications:
    error, err_msge = _verify_common(time_unit, frequency, case_assignment, numeric_agg, xtr_outlier_rem)
    if error:
        raise ValueError(err_msge)
    
    # verify max_k: 
    error, err_msge = ca._verify_max_k(max_k)
    if error: 
        raise ValueError(err_msge)



def _verify_dfr(dfrelations, directly_follows_relations, outcome, time_unit, frequency, case_assignment, plt_type, numeric_agg, xtr_outlier_rem):
    '''
    Checks whether all arguments are correctly specified for the dfr_evol() method.

    '''

    # verify plt_type: 
    if plt_type != 'univariate':
        error, err_msge = ca._verify_typedfr(plt_type)
        if error:
            raise ValueError(err_msge)

    # verify outcome (if plt_type = 'type_outcome):
    if plt_type == 'type_outcome':
        error, err_msge = ca._verify_outcomeCol(outcome)
        if error:
            raise ValueError(err_msge)

    # common verifications:
    error, err_msge = _verify_common(time_unit, frequency, case_assignment, numeric_agg, xtr_outlier_rem)
    if error:
        raise ValueError(err_msge)

    # Verify the passed on dfr's in the directly_follows_relations list:
    error, err_msge = ca._verify_dfrelations(dfrelations, directly_follows_relations)
    if error:
        raise ValueError(err_msge)


def _verify_topKvariants(outcome, time_unit, frequency, case_assignment, plt_type, numeric_agg, max_k, xtr_outlier_rem):
    '''
    Checks whether all arguments are correctly specified for the topK_variants_evol() plotting method.
    '''
    
    # verify plt_type: 
    if plt_type != 'univariate':
        error, err_msge = ca._verify_typevar(plt_type)
        if error:
            raise ValueError(err_msge)

    # verify outcome (if plt_type = 'type_outcome):
    if plt_type == 'type_outcome':
        error, err_msge = ca._verify_outcomeCol(outcome)
        if error:
            raise ValueError(err_msge)

    # common verifications:
    error, err_msge = _verify_common(time_unit, frequency, case_assignment, numeric_agg, xtr_outlier_rem)
    if error:
        raise ValueError(err_msge)
    
    # verify max_k: 
    error, err_msge = ca._verify_max_k(max_k)
    if error: 
        raise ValueError(err_msge)

def _verify_variants(all_vars, variants, outcome, time_unit, frequency, case_assignment, plt_type, numeric_agg, xtr_outlier_rem):
    '''
    Checks whether all arguments are correctly specified for the variants_evol() plotting method.
    '''

    # verify plt_type: 
    if plt_type != 'univariate':
        error, err_msge = ca._verify_typevar(plt_type)
        if error:
            raise ValueError(err_msge)

    # verify outcome (if plt_type = 'type_outcome):
    if plt_type == 'type_outcome':
        error, err_msge = ca._verify_outcomeCol(outcome)
        if error:
            raise ValueError(err_msge)

    # common verifications:
    error, err_msge = _verify_common(time_unit, frequency, case_assignment, numeric_agg, xtr_outlier_rem)
    if error:
        raise ValueError(err_msge)

    # Verify the passed on variants in the 'variants' list:
    error, err_msge = ca._verify_variantslist(all_vars, variants)
    if error:
        raise ValueError(err_msge)

def _verify_topKcatFt(feature, ftr_type, feature_list, outcome, time_unit, frequency, case_assignment, plt_type, numeric_agg, max_k, xtr_outlier_rem):
    '''
    Checks whether all arguments are correctly specified for the topK_categorical_caseftr_evol() or for the topK_categorical_eventftr_evol() plotting methods.

    args:
        - feature       : string; column name of categorical feature to be plotted
        - ftr_type      : string = 'case' or 'event'; specifying whether the categorical feature is a case or an event feature. 
        - feature_list  : list of strings; list of the categorical case or event features names upon initialization of the DynamicLogPlots instance. 
    '''

    # verify plt_type: 
    if plt_type != 'univariate':
        error, err_msge = ca._verify_typeftrs(plt_type)
        if error:
            raise ValueError(err_msge)

    # verify outcome (if plt_type = 'type_outcome):
    if plt_type == 'type_outcome':
        error, err_msge = ca._verify_outcomeCol(outcome)
        if error:
            raise ValueError(err_msge)

    # common verifications:
    error, err_msge = _verify_common(time_unit, frequency, case_assignment, numeric_agg, xtr_outlier_rem)
    if error:
        raise ValueError(err_msge)
    
    # verify max_k: 
    error, err_msge = ca._verify_max_k(max_k)
    if error: 
        raise ValueError(err_msge)
    
    # verify whether feature already in appropriate feature_list
    if feature not in feature_list:
        err_msge = "Column '{}' not yet added to the categorical_{}features list. Add it by means of \
calling the add_categorical_{}ft({}) on your DynamicLogPlots instance.".format(feature, ftr_type, ftr_type, feature)
        raise ValueError(err_msge)

def _verify_numFt(features, ftr_type, feature_list, outcome, time_unit, frequency, case_assignment, 
                  plt_type, numeric_agg, xtr_outlier_rem, numEventFt_transform = None):
    '''
    Checks whether all arguments are correctly specified for the num_casefts_evol() or for the num_eventfts_evol() plotting methods.

    args:
        - features      : list of strings; containing the column names of numerical features to be plotted
        - ftr_type      : string = 'case' or 'event'; specifying whether the numerical features are case or event features. 
        - feature_list  : list of strings; list of the numerical case or event features names upon initialization of the DynamicLogPlots instance. 
    '''
    if type(features) != list:
        err_msge = "The numerical features that need to be plotted should be passed on as a list of strings."
        raise TypeError(err_msge)

    # verify plt_type: 
    if plt_type != 'univariate':
        error, err_msge = ca._verify_typeftrs(plt_type)
        if error:
            raise ValueError(err_msge)

    # verify outcome (if plt_type = 'type_outcome):
    if plt_type == 'type_outcome':
        error, err_msge = ca._verify_outcomeCol(outcome)
        if error:
            raise ValueError(err_msge)

    # common verifications:
    error, err_msge = _verify_common(time_unit, frequency, case_assignment, numeric_agg, xtr_outlier_rem)
    if error:
        raise ValueError(err_msge)
    
    # verify whether the features already contained within the appropriate feature_list
    for feature in features:
        if feature not in feature_list:
            err_msge = "Column '{}' not yet added to the numerical_{}features list. Add it by means of \
calling the add_numerical_{}ft({}) on your DynamicLogPlots instance.".format(feature, ftr_type, ftr_type, feature)
            raise ValueError(err_msge)
    
    # in case of numerical event features, an additional parameter needs to be checked: 
    if ftr_type == 'event':
        error, err_msge = ca._verify_numEventFt_transform(numEventFt_transform)
        if error:
            raise ValueError(err_msge)

def _verify_distinctvars(outcome, time_unit, frequency, case_assignment, plt_type, numeric_agg, xtr_outlier_rem, cases_initialized):
    '''
    Checks whether all arguments are correctly specified for the distinct_variants_evol() and distinct_variants_AdvancedEvol() plotting methods.
    '''
    # verify plt_type: 
    if plt_type != 'univariate':
        error, err_msge = ca._verify_typeDistVars(plt_type)
        if error:
            raise ValueError(err_msge)

    # verify outcome (if plt_type = 'type_outcome):
    if plt_type == 'type_outcome':
        error, err_msge = ca._verify_outcomeCol(outcome)
        if error:
            raise ValueError(err_msge)

    # common verifications:
    error, err_msge = _verify_common(time_unit, frequency, case_assignment, numeric_agg, xtr_outlier_rem)
    if error:
        raise ValueError(err_msge)
    
    # verify cases_initialized:
    error, err_msge = ca._verify_cases_initialized(cases_initialized = cases_initialized)
    if error:
        raise TypeError(err_msge)


def _verify_initial(log, case_id_key, activity_key, timestamp_key, categorical_casefeatures, 
                    numerical_casefeatures, categorical_eventfeatures, numerical_eventfeatures, outcome, start_date, end_date):
    
    # Verify whether the log is actually a pd.DataFrame:
    print("Verification step 1/10", end = '\r')
    if type(log) != pd.core.frame.DataFrame:
        raise TypeError("The log object should be a pandas DataFrame. Convert the log to a pandas DataFrame first.")

    # Verify whether there are no missing values for the case_id_key: 
    print("Verification step 2/10", end = '\r')
    if log[case_id_key].isna().any():
        err_msge = "Column '{}', which should contain the case IDs, contains missing values. Please make sure that every row contains a valid case id."

    # Verify whether the given columns are actually in the passed on log
    print("Verification step 3/10", end = '\r')
    cols = list(log.columns)
    given_cols = [case_id_key, activity_key, timestamp_key] # + categorical_casefeatures + numerical_casefeatures + categorical_eventfeatures + numerical_eventfeatures
    error, err_msge = ca._verify_ColsInLog(cols, given_cols)
    if error:
        raise KeyError(err_msge)
    

    # Verify whether the timestamp_key column is actually a datetime64 column:
    print("Verification step 4/10", end = '\r')
    if not pd.api.types.is_datetime64_any_dtype(log[timestamp_key]):
        raise TypeError("Column '{}', which should contain the timestamps, is not of a datetime64 dtype.".format(timestamp_key))

    # Verify the numerical case features
    print("Verification step 5/10", end = '\r')
    for num_ft in numerical_casefeatures:
        error, err_msge, err_cat = ca._verify_newNumCaFt(log, case_feature = num_ft, case_id_key = case_id_key)
        if error:
            if err_cat == 'type_error':
                raise TypeError(err_msge)
            elif err_cat == 'key_error':
                raise KeyError(err_msge)
            else:
                raise ValueError(err_msge)
    print("Verification step 6/10", end = '\r')
    # Verify the numerical event features:
    for num_ft in numerical_eventfeatures:
        error, err_msge, err_cat = ca._verify_newNumEvFt(log, event_feature = num_ft)
        if error:
            if err_cat == 'type_error':
                raise TypeError(err_msge)
            elif err_cat == 'key_error':
                raise KeyError(err_msge)
            else:
                raise ValueError(err_msge)
    
    # Verify the categorical case features:
    print("Verification step 7/10", end = '\r')
    for cat_ft in categorical_casefeatures:
        error, err_msge, err_cat = ca._verify_newCatCaFt(log, case_feature = cat_ft, case_id_key = case_id_key)
        if error:
            if err_cat == 'key_error':
                raise KeyError(err_msge)
            elif err_cat == 'type_error':
                raise TypeError(err_msge)
            else:
                raise ValueError(err_msge)

    # Verify the categorical event features:
    print("Verification step 8/10", end = '\r')
    for cat_ft in categorical_eventfeatures:
        error, err_msge, err_cat = ca._verify_newCatEvFt(log, event_feature = cat_ft)
        if error:
            if err_cat == 'key_error':
                raise KeyError(err_msge)
            elif err_cat == 'type_error':
                raise TypeError(err_msge)
            else:
                raise ValueError(err_msge)

    # Verify start and / or end date if specified 
    print("Verification step 9/10", end = '\r')
    if start_date or end_date: 
        error, err_msge, err_cat = ca._verify_start_end_date(start_date, end_date)
        if error:
            if err_cat == 'type_error':
                raise TypeError(err_msge)
            else:
                raise ValueError(err_msge)
                
    # Verify outcomn column if specified. 
    print("Verification step 10/10", end = '\r')
    if outcome: #If not None
        error, err_msge = ca._verify_ColsInLog(cols, [outcome])
        if error:
            raise KeyError(err_msge)
        error, err_msge = ca._verify_outcomeCol_initial(log, outcome, case_id_key = case_id_key)
        if error: 
            raise ValueError(err_msge)


# Utility method specific checks 

#   Utility method to verify the to-be added categorical case feature 
def _verify_addCatCaFt(log, case_feature):
    ''' Validity check specifically for verifying whether the to-be added case_feature in the add_categorical_caseft(self, case_feature)
        method is valid. Raises a ValueError if not valid. 

        args:
            - log: pd.DataFrame 
            - case_feature: string containing the column name (in log) of the categorical case feature to-be added. 
    '''
    error, err_msge, err_cat = ca._verify_newCatCaFt(log, case_feature)
    if error:
        if err_cat == 'key_error':
            raise KeyError(err_msge)
        elif err_cat == 'type_error':
            raise TypeError(err_msge)
        else:
            raise ValueError(err_msge)


#   Utility method to verify the to-be added numerical case feature 
def _verify_addNumCaFt(log, case_feature):
    ''' Validity check specifically for verifying whether the to-be added case_feature in the add_numerical_caseft(self, case_feature)
        method is valid. Raises a ValueError if not valid. 

        args:
            - log: pd.DataFrame 
            - case_feature: string containing the column name (in log) of the numerical case feature to-be added. 
    '''
    error, err_msge, err_cat = ca._verify_newNumCaFt(log, case_feature)
    if error:
        if err_cat == 'key_error':
            raise KeyError(err_msge)
        elif err_cat == 'type_error':
            raise TypeError(err_msge)
        else:
            raise ValueError(err_msge)
    
#   Utility method to verify the to-be added categorical event feature 
def _verify_addCatEvFt(log, event_feature):
    ''' Validity check specifically for verifying whether the to-be added event_feature in the add_categorical_eventft(self, event_feature)
        method is valid. Raises a ValueError if not valid. 

        args:
            - log: pd.DataFrame 
            - case_feature: string containing the column name (in log) of the categorical case feature to-be added. 
    '''
    error, err_msge, err_cat = ca._verify_newCatEvFt(log, event_feature)
    if error:
        if err_cat == 'key_error':
            raise KeyError(err_msge)
        else:
            raise TypeError(err_msge)

#   Utility method to verify the to-be added numerical event feature 
def _verify_addNumEvFt(log, event_feature):
    ''' Validity check specifically for verifying whether the to-be added event_feature in the add_numerical_eventft(self, event_feature)
        method is valid. Raises a ValueError if not valid. 

        args:
            - log: pd.DataFrame 
            - event_feature: string containing the column name (in log) of the numerical event feature to-be added. 
    '''
    error, err_msge, err_cat  = ca._verify_newNumEvFt(log, event_feature)
    if error:
        if err_cat == 'key_error':
            raise KeyError(err_msge)
        else:
            raise TypeError(err_msge)

#   Utility method to verify the to-be added outcome column
def _verify_addOutCol(log, outcome):
    ''' Validity check specifically for verifying whether the to-be added event_feature in the add_numerical_eventft(self, event_feature)
        method is valid. Raises a ValueError if not valid. 

        args:
            - log: pd.DataFrame 
            - outcome: string containing the column name (in log) of the outcome feature to be added. 
    '''

    cols = list(log.columns)
    error, err_msge = ca._verify_ColsInLog(cols, [outcome])
    if error:
        raise KeyError(err_msge)
    error, err_msge = ca._verify_outcomeCol_initial(log, outcome, case_id_key = 'case:concept:name')
    if error: 
        raise ValueError(err_msge)


def _verify_select_time_range(start_date, end_date):
    '''Validity check for verifyign the start_date and end_date specification, if any. Specificaly for the 
    select_time_range(start_date, end_date) method. 
    '''

    if start_date or end_date: 
        error, err_msge, err_cat = ca._verify_start_end_date(start_date, end_date)
        if error:
            if err_cat == 'type_error':
                raise TypeError(err_msge)
            else:
                raise ValueError(err_msge)

def _verify_get_DFR_df(max_k, directly_follows_relations, counts, dfrelations):
    """Verify the input parameters of the `get_DFR_df()` method 
    of the `DynamicLogPlots` class. 

    Parameters
    ----------
    max_k : int
    directly_follows_relations : list of tuple
        List of DFRs specified by the user. 
    counts : bool 
        Boolean indicating whether DFR counts should 
        be included. 
    dfrelations: list of tuple
        List of all DFRs present in the event log. 
    """
    # Verify counts:
    error, err_msge = ca._verify_counts_bool(counts)
    if error: 
        raise TypeError(err_msge)

    # Verify max_k 
    if max_k:
        error, err_msge = ca._verify_max_k(max_k)
        if error:
            raise ValueError(err_msge)
    # If max_k = None, directly_follows_relations should be (correctly)
    # specified. 
    else:
        # Check whether directly_follows_relations not None: 
        error, err_msge = ca._verify_dfrlist_None(directly_follows_relations)
        if error: 
            raise ValueError(err_msge)
        # Check whether DFRs exist and / or correctly specified:
        error, err_msge = ca._verify_dfrelations(dfrelations, directly_follows_relations)
        if error: 
            raise ValueError(err_msge)

def _verify_get_var_df(max_k, variants, counts, all_vars):
    """Verify the input parameters of the `get_var_df()` method 
    of the `DynamicLogPlots` class. 

    Parameters
    ----------
    max_k : int
    variants : list of tuple
        List of variants specified by the user. 
    counts : bool 
        Boolean indicating whether DFR counts should 
        be included. 
    all_vars: list of tuple
        List of all variants present in the event log. 
    """ 
    # Verify counts:
    error, err_msge = ca._verify_counts_bool(counts)
    if error: 
        raise TypeError(err_msge)

    # Verify max_k 
    if max_k:
        error, err_msge = ca._verify_max_k(max_k)
        if error:
            raise ValueError(err_msge)
    # If max_k = None, directly_follows_relations should be (correctly)
    # specified. 
    else:
        # Check whether directly_follows_relations not None: 
        error, err_msge = ca._verify_varlist_None(variants)
        if error: 
            raise ValueError(err_msge)
        # Check whether the variants exist and / or correctly specified:
        error, err_msge = ca._verify_variantslist(all_vars, variants)
        if error:
            raise ValueError(err_msge)
    