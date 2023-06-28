import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
import warnings

# Filter out the unnecessary warning message
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information.", category=UserWarning)

def _preprocess_pipeline(log, start_date, end_date):
    ''' Executes the preprocessing pipeline that prepares the given log for plotting. 

        args:
            - log               : pd.DataFrame 
            - event_features    : list of strings, containing the column names of the event features to be preprocessed.
            - start_date        : string "dd/mm/YYYY"
            - end_date          : string "dd/mm/YYYY"
        
        returns:
            - log               : preprocessed pd.DataFrame
    
    '''

    # Verify whether for each case, the events (rows) are arranged in the ascending order of the timestamps.
    # If not, issue a warning. Order automatically fixed.
    log = order_events(log)
    
    # Filter time range if start_date and / or end_date given.
    if start_date or end_date:
        log = select_timerange(log, start_date = start_date, end_date = end_date)

    #Adding periodic timestamps
    log = add_periodic_timestamps(log)

    #Adding case durations (/throughput times):
    log = add_case_duration(log)

    # Adding num_events per case: 
    log = add_num_events(log)

    return log 

# Preprocessing functions called by the preprocessing pipeline:
def select_timerange(log, start_date, end_date):
    """Select only those cases starting after `start_date`, and ending 
    before `end_date`. 

    Parameters
    ----------
    log : pd.DataFrame
        The event log. 
    start_date : str
        Start date as a string. Format "dd/mm/YYYY".
    end_date : str
        End date as a string. Format "dd/mm/YYYY".

    Returns
    -------
    log : pd.DataFrame
        The event log filtered based on the given time range. 
    """

    if start_date:
        start_date = pd.to_datetime(start_date, format = "%d/%m/%Y").tz_localize('UTC')
    else: 
        start_date = log['time:timestamp'].min() #timestamp object 

    if end_date:
        end_date = pd.to_datetime(end_date, format = "%d/%m/%Y").tz_localize('UTC')
    else: 
        end_date = log['time:timestamp'].max() #timestamp object 
        end_date = end_date +  pd.DateOffset(months=1)

    # Get first and last timestamp for each unqiue case 
    first_last = log.pivot_table(values = 'time:timestamp', index = 'case:concept:name', aggfunc = ['min', 'max']).reset_index()
    first_last.columns = first_last.columns.get_level_values(0)
    first_last.columns = ['case:concept:name', '_first_stamp_case', '_last_stamp_case']
    
    # Temporarily add first and last timestamp to log 
    log = log.merge(first_last, on = 'case:concept:name', how = 'left')
    # Filter on case level
    log = log[(log['_first_stamp_case'] >= start_date) & (log['_last_stamp_case'] <= end_date)]
    # Drop temporary columns
    log = log.drop(columns=['_first_stamp_case', '_last_stamp_case'], axis = 1)
    
    return log

def order_events(log):
    ''' Verifies whether for each case, the events (rows) are arranged in the ascending order of the timestamps.
        If not, a warning is issued. The correct ordering is automatically done.
    '''
    local_log = log.copy().reset_index(drop = True)
    case_log = local_log.drop_duplicates(subset = 'case:concept:name').copy()
    case_log = case_log[['case:concept:name']]
    case_log.loc[:, 'int_id'] = [i for i in range(len(case_log))]
    local_log = local_log.merge(case_log, on = 'case:concept:name', how = 'left')
    local_log = local_log.sort_values(['int_id', 'time:timestamp'])
    original_sorted_index = pd.Series(local_log.index.copy())
    local_log.reset_index(drop = True, inplace = True)
    new_sorted_index = pd.Series(local_log.index)
    wrong_order = (new_sorted_index != original_sorted_index).any()

    if wrong_order:
        warning_message = get_warning_message()
        warnings.warn(warning_message)

    local_log = local_log.drop(['int_id'], axis = 1)
    return local_log


def add_periodic_timestamps(log):
    """Preprocesses the event log by adding periodic timestamps for each
    possible case grouping.

    The case grouping determines how each case is assigned to a certain 
    time period. This is jointly determined by both the 'frequency' and 
    'case_assignment' argument present in each plotting method of the 
    'DynamicLogPlots' class. A case feature column is added for each
    ('frequency', 'case_assignment') combination. 

    Parameters
    ----------
    log : pandas.DataFrame
        Current version of the event log. 

    Returns
    -------
    pandas.DataFrame
        The log enhanced with multiple columns. 
    """
    
    def add_two_weekly_timestamp():
        """Compute for each timestamp in the log the two-weekly timestamp
        that represents the beginning of the two-week period in which
        each event occurred. 

        Returns
        -------
        pandas.Series
            Series containing two-weekly timestamps for every event. 
        """
        weekly_timestamps_unique = list(set(list(log['weekly_timestamp'])))
        weekly_timestamps_unique.sort()
        num_weekly_stamps=len(weekly_timestamps_unique)
        ids_retained = [i for i in range(0, num_weekly_stamps, 2)]
        reduced_inds = []
        for ret_idx in ids_retained: 
            red_inds = [ret_idx for _ in range(2)]
            reduced_inds.extend(red_inds)
        reduced_inds= reduced_inds[:num_weekly_stamps]
        reduced_timestamps= []
        for idx in reduced_inds:
            reduced_timestamps.append(weekly_timestamps_unique[idx])
        reduce_dict= dict(zip(weekly_timestamps_unique, reduced_timestamps))
        all_weekly_timestamps = list(log['weekly_timestamp'])
        reduced_timestamps_all = []
        for wkly_stamp in tqdm(all_weekly_timestamps):
            red_stamp = reduce_dict[wkly_stamp]
            reduced_timestamps_all.append(red_stamp)
        two_weekly_timestamp = reduced_timestamps_all
        return two_weekly_timestamp
    def add_halfyearly_timestamp():
        """Compute for each timestamp in the log the half-yearly timestamp
        that represents the beginning of the 6-month period in which
        each event occurred. 

        Returns
        -------
        pandas.Series
            Series containing half_yearly timestamps for every event.
        """

        # Series of integers referring to the month
        month_integer = timestamps.dt.month

        date_minus_3_months = timestamps - pd.DateOffset(months=3)
        minus_3_quarterly_timestamp = date_minus_3_months.dt.to_period("Q").astype('datetime64[ns]')

        second_or_fourth_quarter = (month_integer.isin([4,5,6])) | (month_integer.isin([10,11,12]))

        # Computing half-yearly timestamp. 
        half_yearly_timestamp = np.where(second_or_fourth_quarter, minus_3_quarterly_timestamp, 
                                                log['quarterly_timestamp'])
        return half_yearly_timestamp
    
    def add_first_timestamp(log):
        """Add for every periodic timestamp an additional case feature
        column that indicates the period in which its first event 
        occurred. 

        Parameters
        ----------
        log : pandas.DataFrame
            Current version of the event log. 

        Returns
        -------
        pandas.DataFrame
            Current log object with 'len(periodic_timestamps)' additional cols. 
        """
        extended_PerTS = ['case:concept:name'] + periodic_timestamps
        local_log = log.drop_duplicates(subset = 'case:concept:name')[extended_PerTS].copy()
        new_cols = ['case:concept:name'] + [col + '_firstev' for col in periodic_timestamps]
        local_log.columns = new_cols
        log = log.merge(local_log, on = 'case:concept:name', how = 'left')
        return log 

    def add_last_timestamp(log):
        """Add for every periodic timestamp an additional case feature
        column that indicates the period in which its last event 
        occurred. 

        Parameters
        ----------
        log : pandas.DataFrame
            Current version of the event log. 

        Returns
        -------
        pandas.DataFrame
            Current log object with 'len(periodic_timestamps)' additional cols. 
        """
        extended_PerTS = ['case:concept:name'] + periodic_timestamps
        local_log = log.groupby('case:concept:name', sort= False).last().reset_index()[extended_PerTS].copy()
        new_cols = ['case:concept:name'] + [col + '_lastev' for col in periodic_timestamps]
        local_log.columns = new_cols
        log = log.merge(local_log, on = 'case:concept:name', how = 'left')
        return log 

    def add_maxevents_timestamp(log, time_col):
        """ Add for one periodic timestamp 'time_col' in periodic_timestamps
        an additional case feature (for each case) that stores the 
        'time_col' value (i.e. period) in which most of its events occurred. 

        Parameters
        ----------
        log : pandas.DataFrame
            Current version of the event log. 
        time_col : str
            Indicates the added periodic timestamp column. 

        Returns
        -------
        pd.DataFrame
            Current log object with one additional column. 
        """
        local_log = log[['case:concept:name', 'concept:name', time_col]].copy()
        # For each case: add number of events per period of time_col
        local_log['per_numevs'] = local_log.groupby(['case:concept:name', time_col])['concept:name'].transform('count')
        # For each case: add max 'per_numevs' to each row corresponding to that case
        local_log['max_pernumevs'] = local_log.groupby('case:concept:name')['per_numevs'].transform('max')
        # For each case: filter out the period (time_col stamp) in which most of its events were executed. Ties are broken by taking the first timestamp:
        local_log = local_log[local_log['per_numevs'] == local_log['max_pernumevs']].drop_duplicates(subset= 'case:concept:name')[['case:concept:name', time_col]].copy()
        local_log.columns = ['case:concept:name', time_col+'_maxev']
        # Merging the resulting df back into the original log: 
        log = log.merge(local_log, on = 'case:concept:name', how= 'left')
        return log 
    

    timestamps = log['time:timestamp'].copy()
    # 1-minute interval
    log['1min_timestamp'] = timestamps.dt.floor("1min")

    # 5-minute interval
    log['5min_timestamp'] = timestamps.dt.floor("5min")

    # 10-minute interval
    log['10min_timestamp'] = timestamps.dt.floor("10min")

    # 30-minute interval
    log['30min_timestamp'] = timestamps.dt.floor("30min")

    # 1h interval
    log['hourly_timestamp'] = timestamps.dt.floor("1H")

    # 2-hour interval
    log['2hour_timestamp'] = timestamps.dt.floor("2H")

    # 12-hour interval
    log['12hour_timestamp'] = timestamps.dt.floor("12H")

    # Daily interval:
    log['daily_timestamp'] = timestamps.dt.floor("D")

    # Weekly interval:
    log['weekly_timestamp'] = timestamps.dt.to_period("W").astype('datetime64[ns]')

    # also adding a two_weekly_timestamp: 
    two_weekly_timestamp = add_two_weekly_timestamp()
    log['two_weekly_timestamp'] = two_weekly_timestamp

    # monthly interval
    log['monthly_timestamp'] = timestamps.dt.to_period("M").astype('datetime64[ns]')

    # quarterly interval
    log['quarterly_timestamp'] = timestamps.dt.to_period("Q").astype('datetime64[ns]')

    half_yearly_timestamp = add_halfyearly_timestamp()
    log['half_yearly_timestamp'] = half_yearly_timestamp

    # Add 3 different types of periodic case assignment timestamps: 
    periodic_timestamps = ['1min_timestamp', '5min_timestamp', '10min_timestamp', 
                           '30min_timestamp', 'hourly_timestamp', '2hour_timestamp', 
                           '12hour_timestamp', 'daily_timestamp', 'weekly_timestamp', 
                           'two_weekly_timestamp', 'monthly_timestamp', 
                           'quarterly_timestamp', 'half_yearly_timestamp']
    
    log = add_first_timestamp(log)
    log = add_last_timestamp(log)

    for time_col in periodic_timestamps:
        log = add_maxevents_timestamp(log, time_col)
    
    # Drop the former periodic timestamp columns, since they are not needed anymore: 
    log = log.drop(periodic_timestamps, axis = 1)

    return log 
    
def add_case_duration(log):
    '''
        args: 
            - log:  complete pd.DataFrame 
        returns:
            - log:  pd.DataFrame enhanced with throughput time columns that indicate the case duration. The added columns are:
                    'tt_microseconds', 'tt_milliseconds', 'tt_seconds', 'tt_minutes', 'tt_hours', 'tt_days' and 'tt_weeks'
        
    '''
    local_log = log[['case:concept:name', 'time:timestamp']].copy()
    times = local_log.groupby('case:concept:name', sort= False)['time:timestamp'].agg(['min', 'max']).reset_index()
    times['delta'] = times['max'].copy() - times['min'].copy()
    times['tt_microseconds']  = times['delta'].copy() / pd.Timedelta(microseconds = 1)
    times['tt_milliseconds']  = times['delta'].copy() / pd.Timedelta(milliseconds = 1)
    times['tt_seconds']       = times['delta'].copy() / pd.Timedelta(seconds = 1)
    times['tt_minutes']       = times['delta'].copy() / pd.Timedelta(minutes = 1)
    times['tt_hours']         = times['delta'].copy() / pd.Timedelta(hours = 1)
    times['tt_days']          = times['delta'].copy() / pd.Timedelta(days = 1)
    times['tt_weeks']         = times['delta'].copy() / pd.Timedelta(weeks = 1)
    times = times[['case:concept:name', 'tt_microseconds', 'tt_milliseconds', 'tt_seconds', 'tt_minutes', 'tt_hours', 'tt_days', 'tt_weeks']].copy()
    log = log.merge(times, on = 'case:concept:name', how = 'left')
    return log


def add_num_events(log):
    '''
        args: 
            - log: complete pd.DataFrame 
        returns:
            - log: pd.DataFrame enhanced with the 'num_events' column, indicating for each case how many events a case has. 
    '''
    events_case= log.pivot_table(values='concept:name', index= 'case:concept:name', aggfunc= 'count', fill_value=0).reset_index()
    events_case.columns = ['case:concept:name', 'num_events']
    log = log.merge(events_case, how='left', on='case:concept:name')
    return log 


def get_warning_message():
    warning_message = "In some cases in the given log, the events were not ordered correctly based on their timestamp. \
In the DynamicLogPlots instance' internal representation of the log, this problem is resolved by correctly sorting the events."
    return warning_message