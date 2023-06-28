import pandas as pd
import numpy as np
import datetime as dt

def _verify_ColsInLog(cols, given_cols):
    '''Verifies whether the given column names are actually present in the given log. 

        args:
            - cols: list of the names of all the columns detected in the log
            - given_cols: list of the names of the columns given by the user upon initializing the DynamicLogPlots instance. 

        returns: 
            - error_detected: boolean; True if a given column is not detected in the given log, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    '''
    error_detected, err_msge = False, " "
    for col in given_cols:
        if col not in cols:
            error_detected = True
            err_msge = "'{}': Column with column name '{}' is not found in the given log.".format(col, col)
            return error_detected, err_msge
    return error_detected, err_msge

def _verify_start_end_date(start_date, end_date):
    ''' Verifies whether the start and / or end_date are correclty specified.

        args:
            - start_date: should be a string of the following format 'dd/mm/YYYY'
            - end_date: should be a string of the following format 'dd/mm/YYYY'

        returns:
            - error_detected    : boolean; True if an error is detected in the one of the 2 dates, False otherwise.
            - err_msge          : string; returns an adequate error message if needed. 
            - err_cat           : string = 'type_error' or 'value_error'
    '''
    error_detected, err_msge, err_cat = False, " ", " "
    if start_date:
        try:
            dt.datetime.strptime(start_date, "%d/%m/%Y")
        except ValueError:
            error_detected = True
            err_msge = "'start_date' is incorrectly specified. It shoud be a string of the following format 'dd/mm/YYYY'. \
For example, November 4th 2022 should be specified as the following string: '04/11/2022'"
            err_cat = 'value_error'
            return error_detected, err_msge, err_cat
        except TypeError:
            error_detected = True
            err_msge = "'start_date' is incorrectly specified. It shoud be a string of the following format 'dd/mm/YYYY'. \
For example, November 4th 2022 should be specified as the following string: '04/11/2022'"
            err_cat = 'type_error'
            return error_detected, err_msge, err_cat

    if end_date:
        try:
            dt.datetime.strptime(end_date, "%d/%m/%Y")
        except ValueError:
            error_detected = True
            err_msge = "'end_date' is incorrectly specified. It shoud be a string of the following format 'dd/mm/YYYY'. \
For example, November 4th 2022 should be specified as the following string: '04/11/2022'"
            err_cat = 'value_error'
            return error_detected, err_msge, err_cat
        except TypeError:
            error_detected = True
            err_msge = "'end_date' is incorrectly specified. It shoud be a string of the following format 'dd/mm/YYYY'. \
For example, November 4th 2022 should be specified as the following string: '04/11/2022'"
            err_cat = 'type_error'
            return error_detected, err_msge, err_cat

    # If tests above passed, check whether start_date < end_date
    if start_date and end_date:
        dt.datetime.strptime(start_date, "%d/%m/%Y") >= dt.datetime.strptime(end_date, "%d/%m/%Y")
        error_detected = True
        err_msge = "Both 'start_date' and 'end_date' are correctly specified. However, 'start_date' ('{}') \
comes after 'end_date' ('{}').".format(start_date, end_date)
        err_cat = 'value_error'
        return error_detected, err_msge, err_cat 
    
    return error_detected, err_msge, err_cat





# outcome initialization 
def _verify_outcomeCol_initial(log, outcome, case_id_key):
    ''' Verifies whether the outcome column is correctly formatted. Is only called when a outcome column is specified,
        either when initializing the DynamicLogPlots instance, or when adding an outcome column afterwards with the 
        set_outcomeColumn(outcome) method afterwards. 

        args: 
            - log           : pd.DataFrame
            - outcome       : string referring to the outcome column of log.
            - case_id_key   : string referring to the case_id column of log.

        returns:
            - error_detected: boolean; True if the outcome column is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    '''
    error_detected = False
    err_msge = " "
    # Check whether it is a binary integer column.
    unique_vals = list(log[outcome].unique())
    if not ((unique_vals == [1, 0]) or (unique_vals == [0, 1])):
        error_detected = True
        err_msge = "Column '{}' does not have the right format. The outcome column should be a binary integer column, with the same value (0 or 1) for every event belonging to the same case.".format(outcome)

    else:
        # Check whether it is defined on case level. (Same binary value for each event of the same case)
        num_unique_percase = log.pivot_table(values = outcome, index = case_id_key, aggfunc = pd.Series.nunique)
        if len(num_unique_percase[num_unique_percase[outcome] != 1]) != 0:
            error_detected = True
            err_msge = "Column '{}' does not have the right format. The outcome column should be a binary integer column, with the same value (0 or 1) for every event belonging to the same case.".format(outcome)

    return error_detected, err_msge

# outcome check plotting methods
def _verify_outcomeCol(outcome):
    ''' Verifies whether the outcome column is already specified before including it in a plot. 

        args: 
            - outcome       : string referring to the outcome column of log.

        returns:
            - error_detected: boolean; True if the outcome column is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    '''
    # Should only be called by plotting method specific check functions when plt_type == 'type_outcome'
    error_detected = False
    err_msge = " "
    if outcome == None:
        error_detected = True
        err_msge = "'outcome = None': You want to construct plots of plt_type 'type_outcome', but you have not specified an outcome column. \
To add an outcome column to your DynamicLogPlots instance, you can call the set_outcomeColumn(outcome) method on this instance. E.g. \
<name DynamicLogPlots instance>.set_outcomeColumn('<outcome column name>')."

    return error_detected, err_msge


# plt_type checks 

#   plt_type checks for DFR-related plotting methods
def _verify_typedfr(plt_type):
    ''' Verifies whether the plt_type is correclty specified for the Directly-Follows Relations (dfr) plotting functions.

        args:
            - plt_type: string

        returns:
            - error_detected: boolean; True if plt_type is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    dfr_types = ['univariate', 'type_tt', 'type_events_case', 'type_outcome', 'type_dfr_performance']
    err_detected = False
    err_msge = " "
    if plt_type not in dfr_types:
        err_detected = True
        err_msge = "'{}' is not a valid value for the plt_type argument of this plotting method. The valid options are 'univariate' (default), \
'type_tt', 'type_events_case', 'type_outcome' or 'type_dfr_performance'.".format(plt_type)

    return err_detected, err_msge

#   plt_type checks for variant-related plotting methods 
def _verify_typevar(plt_type):
    ''' Verifies whether the plt_type is correclty specified for the Variants plotting functions.

        args:
            - plt_type: string

        returns:
            - error_detected: boolean; True if plt_type is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    err_detected = False
    err_msge = " "
    if plt_type == 'type_events_case':
        err_detected = True 
        err_msge = "While 'type_events_case' is a valid plot type for other plotting methods, it is not a valid value for the plt_type argument \
of this plotting method. The reason being: the amount of events per case for the same variant remains, by definition, constant over time.  The \
valid options are 'univariate' (default), 'type_tt' or 'type_outcome'."
        return err_detected, err_msge

    var_types = ['univariate', 'type_tt', 'type_outcome']
    if plt_type not in var_types:
        err_detected = True
        err_msge = "'{}' is not a valid value for the plt_type argument of this plotting method. The valid options are 'univariate' (default), 'type_tt' or \
'type_outcome'.".format(plt_type)

    return err_detected, err_msge

#   plt_type checks for case or event feature related plotting methods. 
def _verify_typeftrs(plt_type):
    ''' Verifies whether the plt_type is correclty specified for the (case and event) feature-related plotting functions.

        args:
            - plt_type: string

        returns:
            - error_detected: boolean; True if plt_type is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    ftr_types = ['univariate', 'type_tt', 'type_events_case', 'type_outcome']
    err_detected = False
    err_msge = " "
    if plt_type not in ftr_types:
        err_detected = True
        err_msge = "'{}' is not a valid value for the plt_type argument of this plotting method. The valid options are 'univariate' (default), 'type_tt', \
'type_events_case' or 'type_outcome'.".format(plt_type)

    return err_detected, err_msge

def _verify_typeDistVars(plt_type):
    ''' Verifies whether the plt_type is correclty specified for the distinct variant plotting method.

        args:
            - plt_type: string

        returns:
            - error_detected: boolean; True if plt_type is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    ftr_types = ['univariate', 'type_tt', 'type_events_case', 'type_outcome']
    err_detected = False
    err_msge = " "
    if plt_type not in ftr_types:
        err_detected = True
        err_msge = "'{}' is not a valid value for the plt_type argument of this plotting method. The valid options are 'univariate' (default), 'type_tt', \
'type_events_case' or 'type_outcome'.".format(plt_type)

    return err_detected, err_msge


# time_unit checks:
def _verify_time_unit(time_unit):
    ''' Verifies whether the time_unit is correclty specified for the plotting functions.

        args:
            - time_unit: string

        returns:
            - error_detected: boolean; True if time_unit is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    time_unit_list = ['microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks']
    err_detected = False
    err_msge = " "
    if time_unit not in time_unit_list:
        err_detected = True
        err_msge = "'{}' is not a valid value for the time_unit argument of the plotting methods. The valid options are 'microseconds', \
'milliseconds', 'seconds', 'minutes', 'hours', 'days' (default) or 'weeks'.".format(time_unit)

    return err_detected, err_msge

# frequency checks:
def _verify_frequency(frequency):
    ''' Verifies whether the frequency is correclty specified for the plotting functions.

        args:
            - frequency: string

        returns:
            - error_detected: boolean; True if frequency is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    frequency_list = ['minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly', 
                      '2-hourly', '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 
                      'quarterly', 'half-yearly']
    err_detected = False
    err_msge = " "
    if frequency not in frequency_list:
        err_detected = True
        err_msge = "'{}' is not a valid value for the frequency argument of the plotting methods. The valid options are 'daily', 'weekly' \
(default), 'two_weekly' or 'monthly'.".format(frequency)

    return err_detected, err_msge

# case_assignment checks:
def _verify_case_assignment(case_assignment):
    ''' Verifies whether the case_assignment is correclty specified for the plotting functions.

        args:
            - case_assignment: string

        returns:
            - error_detected: boolean; True if case_assignment is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    case_asg_list = ['first_event', 'last_event', 'max_events']
    err_detected = False
    err_msge = " "
    if case_assignment not in case_asg_list:
        err_detected = True
        err_msge = "'{}' is not a valid value for the case_assignment argument of the plotting methods. The valid options are 'first_event' (default), \
'last_event' or 'max_events'.".format(case_assignment)

    return err_detected, err_msge

# numeric_agg checks:
def _verify_numeric_agg(numeric_agg):
    ''' Verifies whether the numeric_agg is correclty specified for the plotting functions.

        args:
            - numeric_agg: string

        returns:
            - error_detected: boolean; True if case_assignment is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    numagg_list = ['mean', 'median', 'min', 'max', 'std']
    #'mean', 'median', 'min', 'max', 'std'
    err_detected = False
    err_msge = " "
    if numeric_agg not in numagg_list:
        err_detected = True
        err_msge = "'{}' is not a valid value for the numeric_agg argument of the plotting methods. The valid options are 'mean' (default), \
'median', 'min', 'max' or 'std'.".format(numeric_agg)

    return err_detected, err_msge

# Verify numEventFt_transform argument for numerical event feature plotting method 
def _verify_numEventFt_transform(numEventFt_transform):
    numEventFt_transform_options = ['last', 'first', 'mean', 'median', 'sum', 
                                    'prod', 'min', 'max']
    err_detected, err_msge = False, " "
    if numEventFt_transform not in numEventFt_transform_options:
        err_detected = True 
        err_msge = "'{}' is not a valid value for the numEventFt_transform argument of the \
plotting method. The valid options are 'last' (default), 'first', 'mean', 'median', 'sum', \
'prod', 'min' and 'max'."

    return err_detected, err_msge 


# max_k checks:
def _verify_max_k(max_k):
    ''' Verifies whether max_k is correclty specified for the plotting functions.

        args:
            - max_k: integer (or it should be)

        returns:
            - error_detected: boolean; True if max_k is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    err_detected = False
    err_msge = " "

    if type(max_k) != int:
        err_detected = True
        err_msge = "{} is not a valid value for the 'max_k' argument of the plotting methods. 'max_k' (= 10 by default) should be an integer greater than 0.".format(max_k)
    elif max_k < 1:
        err_detected = True
        err_msge = "{} is not a valid value for the 'max_k' argument of the plotting methods. 'max_k' (=10 by default) should be an integer greater than 0.".format(max_k)

    return err_detected, err_msge

# xtr_outlier_rem checks:
def _verify_xtr_outlier_rem(xtr_outlier_rem):
    ''' Verifies whether xtr_outlier_rem is correclty specified for the plotting functions.

        args:
            - xtr_outlier_rem: bool (or it should be)

        returns:
            - error_detected: boolean; True if xtr_outlier_rem is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    err_detected = False
    err_msge = " "

    if type(xtr_outlier_rem) != bool:
        err_detected = True
        err_msge = "{} is not a valid value for the 'xtr_outlier_rem' argument of the plotting methods. \
'xtr_outlier_rem' (= True by default) should be boolean value.".format(xtr_outlier_rem)

    return err_detected, err_msge

def _verify_counts_bool(counts):
    """Verify whether counts parameter is bool dtype. 

    Parameters
    ----------
    counts : bool
    """
    err_detected = False
    err_msge = " "

    if type(counts) != bool:
        err_detected = True
        err_msge = "{} is not a valid value for the 'counts' argument. \
'counts' (= False by default) should be a boolean value.".format(counts)

    return err_detected, err_msge

# cases_initialized:
def _verify_cases_initialized(cases_initialized):
    ''' Verifies whether cases_initialized is correclty specified for the plotting functions.

        args:
            - cases_initialized: bool (or it should be)

        returns:
            - error_detected: boolean; True if xtr_outlier_rem is not well specified, False if no error is detected. 
            - err_msge      : string; returns an adequate error message if needed. 
    
    '''
    err_detected = False
    err_msge = " "

    if type(cases_initialized) != bool:
        err_detected = True
        err_msge = "{} is not a valid value for the 'cases_initialized' argument of the plotting methods. \
'cases_initialized' (= True by default) should be boolean value.".format(cases_initialized)

    return err_detected, err_msge


# Verify the passed on dfr's in the directly_follows_relations list:
def _verify_dfrelations(dfrelations: list, directly_follows_relations: list): 
    ''' Specific verification for the dfr_evol() plotting method. Verifies whether the user-defined Directly-Follows 
        Relations (DFRs), specified in the directly_follows_relations list, are correctly defined and whether they exist in the log. 

        args:
            - dfrelations                   : list of all the dfrelations present in the log.
            - directly_follows_relations    : list of user-defined dfr's to be plotted with the dfr_evol() method. 

        returns:
            - error_detected    : boolean; True if at least one user-defined DFR is specified incorrectly, or when at least one is not even
                                  present in the log.
            - err_msge          : string; returns an adequate error message if needed. 
    
    '''
    err_detected, err_msge = False, " "
    # Verify the passed on dfr's in the directly_follows_relations list:
    if type(directly_follows_relations) != list:
        err_detected = True
        err_msge = "The 'directly_follows_relations' argument is specified incorrectly. It should be specified as a list that contains the Directly-Follows \
Relations (DFRs) that need to be plotted. Note that each DFR (in the 'directly_follows_relations' list) should be represented by a tuple containing the 2 \
corresponding activity strings, e.g. ('activity_a', 'activity_b')."
        return err_detected, err_msge
    for dfrel in directly_follows_relations:
        if dfrel not in dfrelations:
            err_detected = True
            err_msge = "{}: No such Directly-Follows Relation (DFR) was found in the event log. Note that a DFR should be represented by a tuple containing \
the 2 corresponding activity strings. E.g. ('activity_a', 'activity_b').".format(dfrel)
            return err_detected, err_msge
    return err_detected, err_msge

def _verify_dfrlist_None(directly_follows_relations):
    """Verifies whether `directly_follows_relations` specified for the 
    `get_DFR_df()` method of the `DynamicLogPlots` class is not `None`, 
    in case that the method's `max_k` parameter is `None`.

    Parameters
    ----------
    directly_follows_relations : list of tuple
        List of DFRs specified by the user. 

    Returns
    -------
    err_detected : bool 
        `True` if error is detected. 
    err_msge : str
        Appropriate error message if one is detected. 
    """
    err_detected, err_msge = False, " "
    if directly_follows_relations == None:
        err_detected = True 
        err_msge = "Either the `max_k` or `directly_follows_relations` \
parameter should be specified."
    return err_detected, err_msge

def _verify_varlist_None(variants):
    """Verifies whether `variants` specified for the 
    `get_var_df()` method of the `DynamicLogPlots` class is not `None`, 
    in case that the method's `max_k` parameter is `None`.

    Parameters
    ----------
    variants : list of tuple
        List of variants specified by the user. 

    Returns
    -------
    err_detected : bool 
        `True` if error is detected. 
    err_msge : str
        Appropriate error message if one is detected. 
    """
    err_detected, err_msge = False, " "
    if variants == None:
        err_detected = True 
        err_msge = "Either the `max_k` or `variants` \
parameter should be specified."
    return err_detected, err_msge

# Verify the passed on variants in the 'variants' list:
def _verify_variantslist(all_vars: list, variants: list): 
    ''' Specific verification for the variants_evol() plotting method. Verifies whether the user-defined variants, 
        specified in the 'variants' list, are correctly defined and whether they exist in the log. 

        args:
            - all_vars  : list of all the variants present in the log.
            - variants  : list of user-defined variants to be plotted with the dfr_evol() method. 

        returns:
            - error_detected    : boolean; True if at least one user-defined variant is specified incorrectly, or when at least one is not even
                                  present in the log.
            - err_msge          : string; returns an adequate error message if needed. 
    
    '''
    err_detected, err_msge = False, " "
    # Verify the passed on variants in the 'variants' list:
    if type(variants) != list:
        err_detected = True
        err_msge = "The 'variants' argument is specified incorrectly. It should be specified as a list that contains the variants \
that need to be plotted. Note that a variant should be represented by a tuple containing the corresponding activity strings. \
For example, suppose a variant is specified by the sequence of these 3 activities: 'activity_a' - 'activity_b' - \
'activity_c', then this variant should be specified (in the variants list) as ('activity_a', 'activity_b', 'activity_c')."
        return err_detected, err_msge
    for var in variants:
        if var not in all_vars:
            err_detected = True
            err_msge = "{}: No such variant was found in the event log. Note that a variant should be represented by a tuple containing \
the corresponding activity strings. For example, suppose a variant is specified by the sequence of these 3 activities: 'activity_a' - \
'activity_b' - 'activity_c', then this variant should be specified (in the variants list) as ('activity_a', 'activity_b', 'activity_c').".format(var)
            return err_detected, err_msge
    return err_detected, err_msge


# Utility method specific checks 

#   Utility method to verify the to-be added categorical case feature 
def _verify_newCatCaFt(log, case_feature, case_id_key = 'case:concept:name'):
    ''' Validity check specifically for verifying whether the to-be added case_feature in the add_categorical_caseft(self, case_feature)
        method is valid. (Also called upon to verify the categorical case features specified upon initialization of the DynamicLogPlots class.)

        args:
            - log: pd.DataFrame 
            - case_feature: string containing the column name (in log) of the categorical case feature to-be added. 

        returns:
            - error_detected    : boolean; True if an error is detected in the to-be added categorical case feature column, False otherwise.
            - err_msge          : string; returns an adequate error message if needed. 
            - err_cat           : string = 'key_error', 'type_error' or 'value_error'
    '''
    err_detected, err_msge, err_cat = False, " ", " "
    cols = list(log.columns)
    if case_feature not in cols:
        err_detected = True
        err_msge = "'{}': Column with column name '{}' is not found in the given log.".format(case_feature, case_feature)
        err_cat = 'key_error'
        return err_detected, err_msge, err_cat

    is_obj = pd.api.types.is_object_dtype(log[case_feature])
    is_bool = pd.api.types.is_bool_dtype(log[case_feature])
    if not (is_obj or is_bool):
        err_detected = True
        err_msge = "Column '{}' is specified as a categorical case feature, but is not of the correct datatype. The categorical feature \
columns should be of the object dtype or of the boolean dtype.".format(case_feature)
        err_cat = 'type_error'
        return err_detected, err_msge, err_cat
    
    # If it passed the previous validity check, check whether it does not only contain 1 level
    num_levels = len(log[case_feature].unique())
    if num_levels == 1:
        only_value = log[case_feature].unique()[0]
        err_detected = True
        err_msge = "Column '{}' is specified as a categorical case feature, but contains only one value, namely '{}'. Hence, \
categorical case feature '{}' contains no informational value.".format(case_feature, only_value, case_feature)
        err_cat = 'value_error'
        return err_detected, err_msge, err_cat
    
    # Verify whether the case feature actually has only one constant value per case
    num_unique_percase = log.pivot_table(values = case_feature, index = case_id_key, aggfunc = pd.Series.nunique)
    if len(num_unique_percase[num_unique_percase[case_feature] > 1]) > 0:
        err_detected = True 
        err_msge = "Column '{}' is specified as a categorical case feature, but is not a valid case feature. For each individual case, \
a case feature should only have one constant value, while in feature column '{}', (some of) the cases have multiple values for this feature. \
If you are dealing with a categorical event feature, it should be added to categorical_eventfeatures.".format(case_feature, case_feature)
        err_cat = 'value_error'
    
    return err_detected, err_msge, err_cat


#   Utility method to verify the to-be added numerical case feature 
def _verify_newNumCaFt(log, case_feature, case_id_key = 'case:concept:name'):
    ''' Validity check specifically for verifying whether the to-be added case_feature in the add_numerical_caseft(self, case_feature)
        method is valid. (Also called upon to verify the numerical case features specified upon initialization of the DynamicLogPlots class.)

        args:
            - log: pd.DataFrame 
            - case_feature: string containing the column name (in log) of the numerical case feature to-be added. 

        returns:
            - error_detected    : boolean; True if an error is detected in the to-be added numerical case feature column, False otherwise.
            - err_msge          : string; returns an adequate error message if needed.
            - err_cat           : string = 'key_error', 'type_error' or 'value_error'
    '''
    err_detected, err_msge, err_cat = False, " ", " "

    # Verify whether the feature is actually in the log
    cols = list(log.columns)
    if case_feature not in cols:
        err_detected = True
        err_msge = "'{}': Column with column name '{}' is not found in the given log.".format(case_feature, case_feature)
        err_cat = 'key_error'
        return err_detected, err_msge, err_cat

    # Verify whether the numerical feature is actually of a numerical dtype.
    if not pd.api.types.is_numeric_dtype(log[case_feature]):
        err_detected = True 
        err_msge = "Column '{}' is specified as a numerical case feature, but is not of a numeric dtype.".format(case_feature)
        err_cat = 'type_error'
        return err_detected, err_msge, err_cat

    # Verify whether the case feature actually has only one constant value per case
    num_unique_percase = log.pivot_table(values = case_feature, index = case_id_key, aggfunc = pd.Series.nunique)
    if len(num_unique_percase[num_unique_percase[case_feature] > 1]) > 0:
        err_detected = True 
        err_msge = "Column '{}' is specified as a numerical case feature, but is not a valid case feature. For each individual case, \
a case feature should only have one constant value, while in feature column '{}', (some of) the cases have multiple values for this feature. \
If you are dealing with a numerical event feature, it should be added to numerical_eventfeatures.".format(case_feature, case_feature)
        err_cat = 'value_error'

    return err_detected, err_msge, err_cat
    
#   Utility method to verify the to-be added categorical event feature 
def _verify_newCatEvFt(log, event_feature):
    ''' Validity check specifically for verifying whether the to-be added event_feature in the add_categorical_eventft(self, event_feature)
        method is valid. (Also called upon to verify the categorical event features specified upon initialization of the DynamicLogPlots class.)

        args:
            - log: pd.DataFrame 
            - event_feature: string containing the column name (in log) of the categorical event feature to-be added. 

        returns:
            - error_detected    : boolean; True if an error is detected in the to-be added categorical event feature column, False otherwise.
            - err_msge          : string; returns an adequate error message if needed.
            - err_cat           : string = 'key_error' or 'type_error'
    '''
    err_detected, err_msge, err_cat = False, " ", " "
    cols = list(log.columns)
    if event_feature not in cols:
        err_detected = True
        err_msge = "'{}': Column with column name '{}' is not found in the given log.".format(event_feature, event_feature)
        err_cat = 'key_error'
        return err_detected, err_msge, err_cat
    is_obj = pd.api.types.is_object_dtype(log[event_feature])
    is_bool = pd.api.types.is_bool_dtype(log[event_feature])
    if not (is_obj or is_bool):
        err_detected = True 
        err_msge = "Column '{}' is specified as a categorical event feature, but is not of the correct datatype. The categorical feature \
columns should be of the object dtype or of the boolean dtype.".format(event_feature)
        err_cat = 'type_error'
    return err_detected, err_msge, err_cat

#   Utility method to verify the to-be added numerical event feature 
def _verify_newNumEvFt(log, event_feature):
    ''' Validity check specifically for verifying whether the to-be added event_feature in the add_numerical_eventft(self, event_feature)
        method is valid. (Also called upon to verify the numerical event features specified upon initialization of the DynamicLogPlots class.)

        args:
            - log: pd.DataFrame 
            - event_feature: string containing the column name (in log) of the numerical event feature to-be added. 

        returns:
            - error_detected    : boolean; True if an error is detected in the to-be added numerical event feature column, False otherwise.
            - err_msge          : string; returns an adequate error message if needed. 
            - err_cat           : string = 'key_error' or 'type_error'
    '''
    err_detected, err_msge, err_cat = False, " ", " "
    cols = list(log.columns)
    if event_feature not in cols:
        err_detected = True
        err_msge = "'{}': Column with column name '{}' is not found in the given log.".format(event_feature, event_feature)
        err_cat = 'key_error'
        return err_detected, err_msge, err_cat
    if not pd.api.types.is_numeric_dtype(log[event_feature]):
        err_detected = True 
        err_msge = "Column '{}' is specified as a numerical event feature, but is not of a numeric dtype.".format(event_feature)
        err_cat = 'type_error'
    return err_detected, err_msge, err_cat
