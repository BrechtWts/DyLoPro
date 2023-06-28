from DyLoPro.preprocess_utils import _preprocess_pipeline, select_timerange
import DyLoPro.univariate_plots as up
from DyLoPro.plotting_utils import get_variant_case, get_sorted_DFRs, get_filtered_dfr_df, get_filtered_var_df
import DyLoPro.validate_methods as valm
import pandas as pd

    

class DynamicLogPlots():
    def __init__(self, event_log, case_id_key = 'case:concept:name', activity_key = 'concept:name', timestamp_key = 'time:timestamp',
                    categorical_casefeatures = [], numerical_casefeatures = [], categorical_eventfeatures = [], numerical_eventfeatures = [], 
                    start_date = None, end_date = None, outcome = None):
        """Initialize a `DynamicLogPlots` instance by specifying the appropriate arguments.

        After successfully initializing a `DynamicLogPlots` object, all of DyLoPro's 
        plotting functionalities can invoked by calling the appropriate 
        `DynamicLogPlots` methods. 

        Parameters
        ----------
        event_log : pandas.DataFrame
            Event log. Events are regarded as instantaneous.  
        case_id_key : str, optional
            Column name (in `event_log`) containing the case ID. All 
            events pertaining to the same case should share the same 
            unique case ID. By default `'case:concept:name'`. 
        activity_key : str, optional
            Column name (in `event_log`) containing the activity labels. 
            By default `'concept:name'`. 
        timestamp_key : str, optional
            Column name (in `event_log`) containing the timestamps for 
            each event. By default `'time:timestamp'`. Should be of a 
            datetime64 dtype. 
        categorical_casefeatures : list of str, optional
            List of strings containing the column names (in `event_log`) 
            that correspond to categorical case features. All categorical 
            case features for which you wish to analyze the dynamics over 
            time must be specified in this list first, or alternatively, 
            after having already initialized a `DynamicLogPlots` object, 
            with the `add_categorical_caseft(case_feature)` class method. 
            By default `[]`. See Notes for more details on how each 
            categorical case feature column should be formatted. 
        numerical_casefeatures : list of str, optional
            List of strings containing the column names (in `event_log`) 
            that correspond to numerical case features. All numerical 
            case features for which you wish to analyze the dynamics over 
            time must be specified in this list first, or alternatively, 
            after having already initialized a `DynamicLogPlots` object, 
            with the `add_numerical_caseft(case_feature)` class method. 
            By default `[]`. See Notes for more details on how each 
            numerical case feature column should be formatted. 
        categorical_eventfeatures : list of str, optional
            List of strings containing the column names (in `event_log`) 
            that correspond to categorical event features. All 
            categorical event features for which you wish to analyze the 
            dynamics over time must be specified in this list first, or 
            alternatively, after having already initialized a 
            `DynamicLogPlots` object, with the 
            `add_categorical_eventft(event_feature)` class method. By 
            default `[]`. See Notes for more details on how each 
            categorical event feature column should be formatted.
        numerical_eventfeatures : list of str, optional
            List of strings containing the column names (in `event_log`) 
            that correspond to numerical event features. All numerical 
            event features for which you wish to analyze the dynamics 
            over time must be specified in this list first, or 
            alternatively, after having already initialized a 
            `DynamicLogPlots` object, with the 
            `add_numerical_eventft(event_feature)` class method. By 
            default `[]`. See Notes for more details on how each 
            numerical event feature column should be formatted.
        start_date : str, optional
            By default `None`. If specified, only the cases starting 
            after that date will be included in the dynamic profiling. 
            Should be specified in the following format `'dd/mm/YYYY'`. 
            For example, November 4th 2022 should be specified as 
            `start_date='04/11/2022'`.
        end_date : str, optional
            By default `None`. If specified, only the cases ending 
            before that date will be included in the dynamic profiling. 
            Should be specified in the following format `'dd/mm/YYYY'`. 
            For example, November 4th 2022 should be specified as 
            `start_date='04/11/2022'`.
        outcome : str, optional
            Column name (in `event_log`) containing the binary case 
            outcome values (if present). By default `None`. Should be of 
            an integer dtype. See Notes for more details on how an 
            outcome column should be formatted.

        Notes
        -----
        Formatting requirements specified columns in `event_log`: 
        - `categorical_casefeatures` : Every column in `event_log` 
        specified in this list has to be of one of the following dtypes: 
        category, object, boolean. Furthermore, every event (row) 
        pertaining to the same case (i.e. same case ID specified in the 
        `case_id_key` column) should share the exact same value for each 
        case feature. 
        - `numerical_casefeatures` : Every column in `event_log` 
        specified in this list has to be of a numerical dtype. 
        Furthermore, every event (row) pertaining to the same case (i.e. 
        same case ID specified in the `case_id_key` column) should share 
        the exact same value for each case feature. 
        - `categorical_eventfeatures` : Every column in `event_log` 
        specified in this list has to be of one of the following dtypes: 
        category, object, boolean. 
        - `numerical_eventfeatures` : Every column in `event_log` 
        specified in this list has to be of a numerical dtype. 
        - `outcome`: An outcome column in `event_log` should be of an 
        integer dtype, only contain the values `1` (positive cases) and 
        `0` (negative cases). We regard outcome as case outcomes, and 
        hence every event (row) pertaining to the same case (i.e. 
        same case ID specified in the `case_id_key` column) should share 
        the exact same value for the outcome. 
        """
        log = event_log.copy()
        valm._verify_initial(log, case_id_key, activity_key, timestamp_key, categorical_casefeatures, 
                    numerical_casefeatures, categorical_eventfeatures, numerical_eventfeatures, outcome, start_date, end_date)

        if case_id_key != 'case:concept:name':
            log.rename(columns = {case_id_key: 'case:concept:name'}, inplace = True)
        if activity_key != 'concept:name':
            log.rename(columns = {activity_key: 'concept:name'}, inplace = True)
        if timestamp_key != 'time:timestamp':
            log.rename(columns = {timestamp_key: 'time:timestamp'}, inplace = True)
        self.categorical_casefeatures = categorical_casefeatures
        self.numerical_casefeatures = numerical_casefeatures
        self.categorical_eventfeatures = categorical_eventfeatures
        self.numerical_eventfeatures = numerical_eventfeatures
        self.outcome = outcome
        
        # all_event_features = categorical_eventfeatures + numerical_eventfeatures
        print("Preprocessing the data...")
        self.log = _preprocess_pipeline(log = log, start_date= start_date, end_date = end_date) 
        self.original_log = self.log.copy()
        # Getting the list of all Directly-Follows Relations (DFRs) present in the log
        self.dfrelations, self.dfr_df = get_sorted_DFRs(self.log)

        # Getting the (sorted) list of all variants present in the log
        case_variant = get_variant_case(self.log)
        self.all_vars = list(case_variant['variant'].value_counts().index)
        self.variant_df = pd.DataFrame(case_variant['variant'].value_counts()).reset_index()
        self.variant_df.columns = ['variant', 'variant count']



    # Plotting methods
        
    def topK_dfr_evol(self, time_unit = 'days', frequency = 'weekly', case_assignment = 'first_event', plt_type = 'univariate', numeric_agg = 'mean', max_k = 10, xtr_outlier_rem = True):
        """Plot the time series of the requested aggregations for each of the 'max_k' 
        most frequently occurring Directly-Follows Relations (DFRs).

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        holistically combined with different performance measures by specifying the 
        `plt_type` argument. The requested DFR evolutions plotted are arranged in 
        descending order of the number of occurrences of the DFRs.

        For readability, the DFRs are encoded with a number with DFR 1 being the most 
        frequently occurring DFR, and DFR '`max_k`' being the max_k'th most occurring 
        DFR. To retrieve a dataframe that maps these DFR numbers to the actual 
        activity pairs, the `get_DFR_df()` method of the `DynamicLogPlots` class can 
        be called upon. 

        Parameters
        ----------
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome', 'type_dfr_performance'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified 
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        max_k : int, optional
            Only the max_k most frequently occurring DFRs are considered, by default 10. 
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular 
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.

        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - `plt_type` can take one of the following five values:
            - `'univariate'`: For each DFR, the evolutions of the periodically aggregated 
            amount of occurrences per case, as well as the fraction of cases containing
            at least one occurance of of that DFR, are plotted. 
            - `'type_tt'`: For each DFR, next to the univariate plots, also the evolutions of 
            the periodically aggregated Throughput Time (TT) for cases with vs. cases 
            without that DFR are plotted. 
            - `'type_events_case'`: For each DFR, next to the univariate plots, also the evolutions of 
            the periodically aggregated case length (in number of events per case (NEPC)) for cases 
            with vs. cases without that DFR are plotted. 
            - `'type_outcome'`: For each DFR, next to the univariate plots, also the evolutions of 
            the periodic fractions of cases with a positive outcome ('outcome=1') for cases with
            vs. cases without that DFR, are plotted. (Only applicable if an outcome is already 
            specified upon initialization of the `DynamicLogPlots` instance, or with the 
            `set_outcomeColumn(outcome)` method.)
            - `'type_dfr_performance'`: For each DFR, next to the univariate plots, also the 
            evolution of the periodically aggregated DFR performance is plotted. The DFR 
            performance refers to the time elapsed between the first and last activity of
            that DFR. The time unit in which these periodic performance aggregations are 
            expressed, is automatically determined based on their magnitude. __NOTE__ that 
            the `'type_dfr_performance'` representation type is a special case, as it is 
            the only representation type in which cases can deliver more than one measure 
            for a certain DFR. I.e. cases that contain more than one occurrence of a 
            certain DFR will also deliver more than one value for the DFR performance to 
            be aggregated over. 
        - ``numeric_agg`` can take on the following five values:
            - `'mean'` : The requested time series are computed by taking the mean for each 
            time interval. E.g. for `plt_type='univariate'`, for each time interval, 
            and for each of the `max_k` DFRs, the mean amount of occurrences per case 
            is computed. The fraction of cases containing at least one occurrence of a 
            certain DFR is simply computed by, for each time interval, computing the 
            amount of cases with at least one occurrence, and dividing it by the total 
            amount of cases assigned to that time interval. 
            - `'median'` : The requested time series are computed by taking the median for  
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` DFRs, the median amount of occurrences per case, 
            the median TT for cases with at least one occurrence of that DFR, and the 
            median TT for all other cases (without an occurrence of that DFR) are 
            computed. 
            - `'min'` : The requested time series are computed by taking the minimum for 
            each time interval. E.g. for `plt_type='type_events_case'`, for each time interval, 
            and for each of the `max_k` DFRs, the minimum amount of occurrences per case, 
            the minimum NEPC for cases with at least one occurrence of that DFR, and the 
            minimum NEPC for all other cases (without an occurrence of that DFR) are 
            computed. 
            - `'max'` : The requested time series are computed by taking the maximum for 
            each time interval. E.g. for `plt_type='type_dfr_performance'`, for each 
            time interval, and for each of the `max_k` DFRs, the maximum amount of 
            occurrences per case and the maximum DFR performance (i.e. the maximum 
            time elapsed between the occurrence of the first and second activity of 
            that DFR) are computed. 
            - `'std'` : The requested time series are computed by taking the standard 
            deviation for each time interval. E.g. for `plt_type='type_dfr_performance'`, 
            for each time interval, and for each of the `max_k` DFRs, the standard deviation 
            of amount of occurrences per case as and the standard deviation of the DFR 
            performances are computed. 
        """
        valm._verify_topKdfr(outcome = self.outcome, time_unit = time_unit, frequency= frequency, 
                            case_assignment= case_assignment, plt_type= plt_type, numeric_agg= numeric_agg,
                             max_k= max_k, xtr_outlier_rem = xtr_outlier_rem)
        top_k_dfr = self.dfrelations[:max_k]
        up.topK_dfr_evol(log = self.log, top_k_dfr = top_k_dfr, outcome = self.outcome, time_unit = time_unit, frequency = frequency, case_assignment = case_assignment,
                        type = plt_type, numeric_agg = numeric_agg, max_k = max_k, xtr_outlier_rem = xtr_outlier_rem)
    
    def dfr_evol(self, directly_follows_relations, time_unit='days', frequency='weekly', case_assignment = 'first_event', plt_type= 'univariate', numeric_agg= 'mean', xtr_outlier_rem=True):
        """Plot the time series of the requested aggregations for the 
        Directly-Follows Relations (DFRs) specified in the 'directly_follows_relations' list.

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        holistically combined with different performance measures by specifying the 
        `plt_type` argument. The requested DFR evolutions plotted are arranged in 
        descending order of the number of occurrences of the DFRs.

        For readability, the DFRs in `directly_follows_relations` are encoded with a 
        number, with DFR 1 being the most frequently occurring DFR, DFR 2 being 
        the second most occurring DFR, and so on. To retrieve a dataframe that maps these 
        DFR numbers to the actual activity pairs, the `get_DFR_df()` method of the 
        `DynamicLogPlots` class can be called upon. 

        Parameters
        ----------
        directly_follows_relations : list of tuple
            The DFRs for which the requested time series will be plotted. Each DFR needs to 
            be specified as a tuple that contains 2 strings, referring to the 2 activities in 
            the DFR, e.g. ('activity_a', 'activity_b').
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome', 'type_dfr_performance'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular  
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.

        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - `plt_type` can take one of the following five values:
            - `'univariate'`: For each DFR, the evolutions of the periodically aggregated 
            amount of occurrences per case, as well as the fraction of cases containing
            at least one occurance of of that DFR, are plotted. 
            - `'type_tt'`: For each DFR, next to the univariate plots, also the evolutions of 
            the periodically aggregated Throughput Time (TT) for cases with vs. cases 
            without that DFR are plotted. 
            - `'type_events_case'`: For each DFR, next to the univariate plots, also the evolutions of 
            the periodically aggregated case length (in number of events per case (NEPC)) for cases 
            with vs. cases without that DFR are plotted. 
            - `'type_outcome'`: For each DFR, next to the univariate plots, also the evolutions of 
            the periodic fractions of cases with a positive outcome ('outcome=1') for cases with
            vs. cases without that DFR, are plotted. (Only applicable if an outcome is already 
            specified upon initialization of the `DynamicLogPlots` instance, or with the 
            `set_outcomeColumn(outcome)` method.)
            - `'type_dfr_performance'`: For each DFR, next to the univariate plots, also the 
            evolution of the periodically aggregated DFR performance is plotted. The DFR 
            performance refers to the time elapsed between the first and last activity of
            that DFR. The time unit in which these periodic performance aggregations are 
            expressed, is automatically determined based on their magnitude. __NOTE__ that 
            the `'type_dfr_performance'` representation type is a special case, as it is 
            the only representation type in which cases can deliver more than one measure 
            for a certain DFR. I.e. cases that contain more than one occurrence of a 
            certain DFR will also deliver more than one value for the DFR performance to 
            be aggregated over. 
        - ``numeric_agg`` can take on the following five values:
            - `'mean'` : The requested time series are computed by taking the mean for each 
            time interval. E.g. for `plt_type='univariate'`, for each time interval, 
            and for each of the DFRs specified in `directly_follows_relations`, the 
            mean amount of occurrences per case is computed. The fraction of cases 
            containing at least one occurrence of a certain DFR is simply computed by, 
            for each time interval, computing the amount of cases with at least one 
            occurrence, and dividing it by the total amount of cases assigned to 
            that time interval. 
            - `'median'` : The requested time series are computed by taking the median for  
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the DFRs specified in `directly_follows_relations`, the median 
            amount of occurrences per case, the median TT for cases with at least one 
            occurrence of that DFR, and the median TT for all other cases 
            (without an occurrence of that DFR) are computed. 
            - `'min'` : The requested time series are computed by taking the minimum for 
            each time interval. E.g. for `plt_type='type_events_case'`, for each time interval, 
            and for each of the DFRs specified in `directly_follows_relations`, the 
            minimum amount of occurrences per case, the minimum NEPC for cases with 
            at least one occurrence of that DFR, and the minimum NEPC for all other 
            cases (without an occurrence of that DFR) are computed. 
            - `'max'` : The requested time series are computed by taking the maximum for 
            each time interval. E.g. for `plt_type='type_dfr_performance'`, for each 
            time interval, and for each of the DFRs specified in `directly_follows_relations`, 
            the maximum amount of occurrences per case and the maximum DFR performance 
            (i.e. the maximum time elapsed between the occurrence of the first and second 
            activity of that DFR) are computed. 
            - `'std'` : The requested time series are computed by taking the standard 
            deviation for each time interval. E.g. for `plt_type='type_dfr_performance'`, 
            for each time interval, and for each of the DFRs specified in `directly_follows_relations`, 
            the standard deviation of amount of occurrences per case as and the standard 
            deviation of the DFR performances are computed. 
        """

        valm._verify_dfr(dfrelations = self.dfrelations, directly_follows_relations = directly_follows_relations, outcome = self.outcome, 
                        time_unit = time_unit, frequency = frequency, case_assignment = case_assignment, plt_type = plt_type, 
                        numeric_agg = numeric_agg, xtr_outlier_rem= xtr_outlier_rem)

        up.dfr_evol(   log = self.log, directly_follows_relations = directly_follows_relations, outcome = self.outcome, time_unit = time_unit, frequency = frequency,
                    case_assignment = case_assignment, type = plt_type, numeric_agg = numeric_agg, xtr_outlier_rem = xtr_outlier_rem)


    def topK_variants_evol(self, time_unit='days', frequency='weekly', case_assignment = 'first_event', plt_type= 'univariate', numeric_agg= 'mean', max_k= 10, xtr_outlier_rem = True):
        """Plot the time series of the requested aggregations for each of the 'max_k' 
        most frequently occurring variants.

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        holistically combined with different performance measures by specifying the 
        `plt_type` argument. The requested variant time series plotted are arranged in 
        descending order of the number of occurrences of the variants.

        For readability, the variants are encoded with a number, with variant 1 being the most 
        frequently occurring variant, and variant '`max_k`' being the max_k'th most occurring 
        variant. To retrieve a dataframe that maps these variant numbers to the actual 
        activity sequences, the `get_var_df()` method of the `DynamicLogPlots` class can 
        be called upon. 

        Parameters
        ----------
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_outcome'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        max_k : int, optional
            Only the max_k most frequently occurring variants are considered, by default 10. 
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular  
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.

        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - `plt_type` can take one of the following three values:
            - `'univariate'`: For each variant, plots the evolution of the periodic fraction
            of cases for accounted for by that variant. 
            - `'type_tt'`: For each variant, next to the univariate plots, also the evolutions of 
            the periodically aggregated Throughput Time (TT) for cases belonging to that variant vs. 
            all other cases (not belonging to that variant), are plotted. 
            - `'type_outcome'`: For each variant, next to the univariate plots, also the evolutions 
            of the periodic fractions of cases with a positive outcome ('outcome=1') for cases 
            belonging to that variant vs. all other cases (not belonging to that variant),
            are plotted. (Only applicable if an outcome is already specified upon initialization 
            of the `DynamicLogPlots` instance, or with the `set_outcomeColumn(outcome)` method.)
        - ``numeric_agg`` can take on the following five values:
            - `'mean'` : The requested time series are computed by taking the mean for each 
            time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` variants, the mean TT for cases belonging to 
            that variant, and the mean TT for all other cases (belonging to other 
            variants) are computed. 
            - `'median'` : The requested time series are computed by taking the median for  
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` variants, the median TT for cases belonging to 
            that variant, and the median TT for all other cases (belonging to other 
            variants) are computed. 
            - `'min'` : The requested time series are computed by taking the minimum for 
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` variants, the minimum TT among cases belonging 
            to that variant, and the minimum TT among all other cases (belonging to other 
            variants) are computed. 
            - `'max'` : The requested time series are computed by taking the maximum for 
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` variants, the maximum TT among cases belonging 
            to that variant, and the maximum TT among all other cases (belonging to other 
            variants) are computed. 
            - `'std'` : The requested time series are computed by taking the standard 
            deviation for each time interval. E.g. for `plt_type='type_tt'`, 
            for each time interval, and for each of the `max_k` variants, the standard  
            deviation of the TT for cases belonging to that variant, and the standard 
            deviation of the TT for all other cases (belonging to other variants) are computed. 
        """
        valm._verify_topKvariants(outcome = self.outcome, time_unit = time_unit, frequency = frequency, 
                                    case_assignment = case_assignment, plt_type = plt_type, numeric_agg = numeric_agg,
                                    max_k = max_k, xtr_outlier_rem = xtr_outlier_rem)
        top_k_vars = self.all_vars[:max_k]
        up.topK_variants_evol(log = self.log, top_k_vars = top_k_vars, outcome = self.outcome, time_unit= time_unit, frequency= frequency, 
                            case_assignment = case_assignment, type= plt_type, numeric_agg= numeric_agg, max_k= max_k, xtr_outlier_rem = xtr_outlier_rem)

    def variants_evol(self, variants, time_unit='days', frequency='weekly', case_assignment = 'first_event', plt_type= 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True):
        """Plot the time series of the requested aggregations for the 
        variants specified in the 'variants' list.

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        holistically combined with different performance measures by specifying the 
        `plt_type` argument. The requested variant time series plotted are arranged in 
        descending order of the number of occurrences of the variants.

        For readability, the variants are encoded with a number, with variant 1 being the most 
        frequently occurring variant, variant 2 being the second most occurring 
        variant, and so on. To retrieve a dataframe that maps these variant numbers to the actual 
        activity sequences, the `get_var_df()` method of the `DynamicLogPlots` class can 
        be called upon. 

        Parameters
        ----------
        variants : list of tuple
            The variants for which the requested time series will be plotted. Each variant needs to 
            be specified as a tuple that contains N strings, referring to the N activities that 
            constitute that variant. 
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_outcome'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular  
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.

        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - `plt_type` can take one of the following three values:
            - `'univariate'`: For each variant, plots the evolution of the periodic fraction
            of cases accounted for by that variant. 
            - `'type_tt'`: For each variant, next to the univariate plots, also the evolutions of 
            the periodically aggregated Throughput Time (TT) for cases belonging to that variant vs. 
            all other cases (not belonging to that variant), are plotted. 
            - `'type_outcome'`: For each variant, next to the univariate plots, also the evolutions 
            of the periodic fractions of cases with a positive outcome ('outcome=1') for cases 
            belonging to that variant vs. all other cases (not belonging to that variant),
            are plotted. (Only applicable if an outcome is already specified upon initialization 
            of the `DynamicLogPlots` instance, or with the `set_outcomeColumn(outcome)` method.)
        - ``numeric_agg`` can take on the following five values:
            - `'mean'` : The requested time series are computed by taking the mean for each 
            time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each variant specified in `variants`, the mean TT for cases belonging to 
            that variant, and the mean TT for all other cases (belonging to other 
            variants) are computed. 
            - `'median'` : The requested time series are computed by taking the median for  
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each variant specified in `variants`, the median TT for cases belonging to 
            that variant, and the median TT for all other cases (belonging to other 
            variants) are computed. 
            - `'min'` : The requested time series are computed by taking the minimum for 
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each variant specified in `variants`, the minimum TT among cases belonging 
            to that variant, and the minimum TT among all other cases (belonging to other 
            variants) are computed. 
            - `'max'` : The requested time series are computed by taking the maximum for 
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each variant specified in `variants`, the maximum TT among cases belonging 
            to that variant, and the maximum TT among all other cases (belonging to other 
            variants) are computed. 
            - `'std'` : The requested time series are computed by taking the standard 
            deviation for each time interval. E.g. for `plt_type='type_tt'`, 
            for each time interval, and for each variant specified in `variants`, the standard  
            deviation of the TT for cases belonging to that variant, and the standard 
            deviation of the TT for all other cases (belonging to other variants) are computed. 
        """
        valm._verify_variants(all_vars = self.all_vars, variants = variants, outcome = self.outcome, time_unit = time_unit, 
                                frequency = frequency, case_assignment = case_assignment, plt_type = plt_type, 
                                numeric_agg = numeric_agg, xtr_outlier_rem = xtr_outlier_rem)

        up.variants_evol(log = self.log, variants = variants, outcome = self.outcome, time_unit = time_unit,
                            frequency = frequency, case_assignment = case_assignment, type= plt_type, 
                            numeric_agg= numeric_agg, xtr_outlier_rem = xtr_outlier_rem)

    def topK_categorical_caseftr_evol(self, case_feature, time_unit = 'days', frequency = 'weekly', case_assignment = 'first_event', plt_type = 'univariate', numeric_agg = 'mean', max_k = 10, xtr_outlier_rem = True):
        """Plot the time series of the requested aggregations for each of the 'max_k' 
        most frequently occurring levels of categorical case feature 'case_feature'.
        If 'max_k' is greater than or equal to the cardinality of that feature, the 
        requested time series for all levels are plotted. 

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        holistically combined with different performance measures by specifying the 
        `plt_type` argument. The requested level time series plotted are arranged in 
        descending order of the number of occurrences of the levels. 

        Parameters
        ----------
        case_feature : str
            Column name of the categorical case feature. The case feature already has to be 
            specified, either upon initalization of the `DynamicLogPlots` object, or with 
            the `.add_categorical_caseft(case_feature)` class method. 
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        max_k : int, optional
            Only the max_k most frequently occurring levels of 'case_feature' are considered, 
            by default 10.
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular  
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.

        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - `plt_type` can take one of the following four values:
            - `'univariate'`: For each level, plots the evolution of the periodic fraction
            of cases for which `case_feature='level'`. 
            - `'type_tt'`: For each level, next to the univariate plots, also the evolutions of 
            the periodically aggregated Throughput Time (TT) for cases with `case_feature='level'` vs. 
            all other cases (`case_feature!='level'`), are plotted. 
            - `'type_events_case'`: For each level, next to the univariate plots, also the evolutions of 
            the periodically aggregated case length (in Number of Events Per Case (NEPC)) for cases 
            with `case_feature='level'` vs. all other cases (`case_feature!='level'`), are plotted. 
            - `'type_outcome'`: For each level, next to the univariate plots, also the evolutions 
            of the periodic fractions of cases with a positive outcome ('outcome=1') for cases 
            with `case_feature='level'` vs. all other cases (`case_feature!='level'`), are plotted. 
            (Only applicable if an outcome is already specified upon initialization 
            of the `DynamicLogPlots` instance, or with the `set_outcomeColumn(outcome)` method.)
        - ``numeric_agg`` can take on the following five values:
            - `'mean'` : The requested time series are computed by taking the mean for each 
            time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` levels of `case_feature`, the mean TT for cases with 
            `case_feature='level'`, and the mean TT for all other cases (with 
            `case_feature!='level'`) are computed. 
            - `'median'` : The requested time series are computed by taking the median for  
            each time interval. E.g. for `plt_type='type_events_case'`, for each time interval, 
            and for each of the `max_k` levels of `case_feature`, the median NEPC for cases with 
            `case_feature='level'`, and the median NEPC for all other cases (with 
            `case_feature!='level'`) are computed. 
            - `'min'` : The requested time series are computed by taking the minimum for 
            each time interval. E.g. for `plt_type='type_events_case'`, for each time interval, 
            and for each of the `max_k` levels of `case_feature`, the minimum NEPC among cases with 
            `case_feature='level'`, and the minimum NEPC among all other cases (with 
            `case_feature!='level'`) are computed. 
            - `'max'` : The requested time series are computed by taking the maximum for 
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` levels of `case_feature`, the maximum TT among cases 
            with `case_feature='level'`, and the maximum TT among all other cases (with 
            `case_feature!='level'`) are computed. 
            - `'std'` : The requested time series are computed by taking the standard 
            deviation for each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` levels of `case_feature`, the standard deviation of the 
            TT among cases with `case_feature='level'`, and the standard deviation of the TT of all 
            other cases (with `case_feature!='level'`) are computed. 
        """

        valm._verify_topKcatFt(feature = case_feature, ftr_type = 'case', feature_list = self.categorical_casefeatures, 
                                    outcome = self.outcome, time_unit = time_unit, frequency = frequency, case_assignment = case_assignment, 
                                    plt_type = plt_type, numeric_agg = numeric_agg, max_k = max_k, xtr_outlier_rem = xtr_outlier_rem)

        up.topK_categorical_caseftr_evol(log = self.log, case_feature = case_feature, outcome = self.outcome, time_unit = time_unit, 
                                        frequency = frequency, case_assignment = case_assignment, type = plt_type, 
                                        numeric_agg = numeric_agg, max_k = max_k, xtr_outlier_rem = xtr_outlier_rem)


    def num_casefts_evol(self, numeric_case_list, time_unit='days', frequency='weekly', case_assignment = 'first_event', plt_type = 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True):
        """ Plots the time series of the requested aggregations over time for the numerical 
        event features specified in the 'numeric_case_list' argument.

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        holistically combined with different performance measures by specifying the 
        `plt_type` argument.

        Parameters
        ----------
        numeric_case_list : list of str
            Column names of the numerical case features. All case features already have to be 
            specified, either upon initalization of the `DynamicLogPlots` object, or with 
            the `.add_numerical_caseft(case_feature)` class method. 
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular  
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.

        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - `plt_type` can take one of the following four values:
            - `'univariate'`: For each numerical case feature, plots the evolution of its 
            periodical aggregations. 
            - `'type_tt'`: For each numerical case feature, next to the univariate plots, also the
            evolution of the periodically aggregated ratio of Throughput Time (TT) needed per 
            unit of that feature is plotted. The time unit of these periodic ratio aggregations 
            (time unit / unit of feature) is automatically determined based on their magnitude. 
            - `'type_events_case'`: For each numerical case feature, next to the univariate plots, 
            also the evolution of the periodically aggregated ratio of Number of Events Per Case 
            (NEPC) needed per 10^(x) units of that feature is plotted. The exponent 'x' is automatically 
            determined based on their magnitude. 
            - `'type_outcome'`: For each numerical case feature, next to the univariate plots, also 
            the two evolutions of the feature's periodical aggregations for cases with a positive 
            outcome ('outcome=1') vs. cases with a negative outcome ('outcome=1') are plotted. 
            (Only applicable if an outcome is already specified upon initialization 
            of the `DynamicLogPlots` instance, or with the `set_outcomeColumn(outcome)` method.)
        - ``numeric_agg`` can take on the following five values:
            - `'mean'` : The requested time series are computed by taking the mean for each 
            time interval. E.g. for `plt_type='univariate'`, for each time interval, 
            and for each numeric case feature specified in `numeric_case_list`, the mean 
            of all case feature values is is computed. 
            - `'median'` : The requested time series are computed by taking the median for 
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each numeric case feature specified in `numeric_case_list`, the median 
            of all case feature values as well as the median of all TT ratios are computed. 
            - `'min'` : The requested time series are computed by taking the minimum for 
            each time interval. E.g. for `plt_type='type_events_case'`, for each time 
            interval, and for each numeric case feature specified in `numeric_case_list`, 
            both the minimum of all case feature values and the minimum of all NEPC ratios 
            are computed. 
            - `'max'` : The requested time series are computed by taking the maximum for 
            each time interval. E.g. for `plt_type='type_outcome'`, for each time 
            interval, and for each numeric case feature specified in `numeric_case_list`, 
            the maximum of all case feature values, the maximum of all case feature values 
            for cases with a positive outcome and the maximum of all values for cases 
            with a negative outcome are computed.
            - `'std'` : The requested time series are computed by taking the standard 
            deviation for each time interval. E.g. for `plt_type='type_tt'`, for each 
            time interval, and for each numeric case feature specified in `numeric_case_list`, 
            the standard deviation of all case feature values as well as the standard deviation 
            of all TT ratios are computed. 
        """

        valm._verify_numFt(features = numeric_case_list, ftr_type = 'case', feature_list = self.numerical_casefeatures, 
                            outcome = self.outcome, time_unit = time_unit, frequency = frequency, 
                            case_assignment= case_assignment, plt_type = plt_type, numeric_agg = numeric_agg, xtr_outlier_rem= xtr_outlier_rem)
        up.num_casefts_evol(log = self.log, numeric_case_list = numeric_case_list, outcome = self.outcome, time_unit = time_unit, 
                                frequency = frequency, case_assignment = case_assignment, type = plt_type, 
                                numeric_agg= numeric_agg, xtr_outlier_rem= xtr_outlier_rem)


    def topK_categorical_eventftr_evol(self, event_feature, time_unit = 'days', frequency = 'weekly', case_assignment = 'first_event', plt_type = 'univariate', numeric_agg = 'mean', max_k = 10, xtr_outlier_rem = True):
        """Plot the time series of the requested aggregations for each of the 'max_k' 
        most frequently occurring levels of categorical event feature 'event_feature'.
        If 'max_k' is greater than or equal to the cardinality of that feature, the 
        requested time series for all levels are plotted. 

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        holistically combined with different performance measures by specifying the 
        `plt_type` argument. The requested level time series plotted are arranged in 
        descending order of the number of occurrences of each level.

        Parameters
        ----------
        event_feature : str
            Column name of the categorical event feature. The event feature already has to be 
            specified, either upon initalization of the `DynamicLogPlots` object, or with 
            the `.add_categorical_eventft(event_feature)` class method. 
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        max_k : int, optional
            Only the max_k most frequently occurring levels of 'event_feature' are considered, 
            by default 10.
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular  
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.

        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - `plt_type` can take one of the following four values:
            - `'univariate'`: For each level, plots the evolution of the periodic fraction
            of cases for which . 
            - `'type_tt'`: For each level, next to the univariate plots, also the evolutions of 
            the periodically aggregated Throughput Time (TT) for cases with at least one  
            occurrence of  vs. all other cases (`event_feature!='level'` for all 
            events), are plotted. 
            - `'type_events_case'`: For each level, next to the univariate plots, also the evolutions of 
            the periodically aggregated case length (in Number of Events Per Case (NEPC)) for cases 
            with at least one occurrence of  vs. all other cases 
            (`event_feature!='level'` for all events), are plotted. 
            - `'type_outcome'`: For each level, next to the univariate plots, also the evolutions 
            of the periodic fractions of cases with a positive outcome ('outcome=1') for cases 
            with at least one occurrence of `event_feature='level'` vs. all other cases 
            (`event_feature!='level'` for all events), are plotted. (Only applicable if an outcome 
            is already specified upon initialization of the `DynamicLogPlots` instance, or with 
            the `set_outcomeColumn(outcome)` method.)
        - ``numeric_agg`` can take on the following five values:
            - `'mean'` : The requested time series are computed by taking the mean for each 
            time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` levels of `event_feature`, the mean TT for cases with 
            at least one occurrence of `event_feature='level'`, and the mean TT for all other cases 
            (with `event_feature!='level'` for all events) are computed. 
            - `'median'` : The requested time series are computed by taking the median for  
            each time interval. E.g. for `plt_type='type_events_case'`, for each time interval, 
            and for each of the `max_k` levels of `event_feature`, the median NEPC for cases with 
            at least one occurrence of `event_feature='level'`, and the median NEPC for all other cases  
            (with `event_feature!='level'` for all events) are computed. 
            - `'min'` : The requested time series are computed by taking the minimum for 
            each time interval. E.g. for `plt_type='type_events_case'`, for each time interval, 
            and for each of the `max_k` levels of `event_feature`, the minimum NEPC among cases with 
            at least one occurrence of `event_feature='level'`, and the minimum NEPC among all other 
            cases (with `event_feature!='level'` for all events) are computed. 
            - `'max'` : The requested time series are computed by taking the maximum for 
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` levels of `event_feature`, the maximum TT among cases 
            with at least one occurrence of `event_feature='level'`, and the maximum TT among all 
            other cases (with `event_feature!='level'` for all events) are computed. 
            - `'std'` : The requested time series are computed by taking the standard 
            deviation for each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each of the `max_k` levels of `event_feature`, the standard deviation of the 
            TT among cases with at least one occurrence of `event_feature='level'`, and the standard 
            deviation of the TT of all other cases (with `event_feature!='level'` for all events) are 
            computed. 
        """
        valm._verify_topKcatFt(feature = event_feature, ftr_type = 'event', feature_list = self.categorical_eventfeatures,
                                outcome = self.outcome, time_unit = time_unit, frequency = frequency, case_assignment = case_assignment,
                                plt_type = plt_type, numeric_agg = numeric_agg, max_k = max_k, xtr_outlier_rem = xtr_outlier_rem)

        up.topK_categorical_eventftr_evol(log = self.log, event_feature = event_feature, outcome = self.outcome, time_unit = time_unit,
                                frequency = frequency, case_assignment = case_assignment, plt_type = plt_type, numeric_agg = numeric_agg,
                                max_k = max_k, xtr_outlier_rem = xtr_outlier_rem)


    def num_eventfts_evol(self, numeric_event_list, time_unit = 'days', frequency = 'weekly', case_assignment = 'first_event', 
                          plt_type = 'univariate', numeric_agg = 'mean', xtr_outlier_rem = True, numEventFt_transform = 'last'):
        """Plot the evolution of the requested aggregations over time for the numerical 
        event features specified in the numeric_event_list argument. 

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        holistically combined with different performance measures by specifying the 
        `plt_type` argument. 

        In contrast to the numerical case features, numerical event features can take on 
        different values for one and the same case. Therefore, to visualize case-level 
        characteristics over time, an additional abstraction method is needed to project a 
        trace's sequence of numerical event feature values to a single numeric value. This
        transformation can be specified by means of the `numEventFt_transform` parameter. 
        
        Parameters
        ----------
        numeric_event_list : list of str
            Column names of the numerical event features. All event features already have to be 
            specified, either upon initalization of the `DynamicLogPlots` object, or with 
            the `.add_numerical_eventft(event_feature)` class method. 
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular  
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.
        numEventFt_transform : {'last', 'first', 'mean', 'median', 'sum', 'prod', 'min', 'max'}
            If any numeric event features contained in 'event_features', `numEventFt_transform` 
            determines the way in which these numerical event features are transformed to the 
            case level. By default `'last'`. For a more detailed explanation of the different 
            `numEventFt_transform` options, see Notes.

        
        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - ``plt_type`` can take one of the following four values:
            - `'univariate'`: For each numerical event feature, plots the evolution of its 
            periodical aggregations. 
            - `'type_tt'`: For each numerical event feature, next to the univariate plots, also the
            evolution of the periodically aggregated ratio of Throughput Time (TT) needed per 
            unit of that feature is plotted. The time unit of these periodic ratio aggregations 
            (time unit / unit of feature) is automatically determined based on their magnitude. 
            - `'type_events_case'`: For each numerical event feature, next to the univariate plots, 
            also the evolution of the periodically aggregated ratio of Number of Events Per Case 
            (NEPC) needed per 10^(x) units of that feature is plotted. The exponent 'x' is automatically 
            determined based on their magnitude. 
            - `'type_outcome'`: For each numerical event feature, next to the univariate plots, also 
            the two evolutions of the feature's periodical aggregations for cases with a positive 
            outcome ('outcome=1') vs. cases with a negative outcome ('outcome=1') are plotted. 
            (Only applicable if an outcome is already specified upon initialization 
            of the `DynamicLogPlots` instance, or with the `set_outcomeColumn(outcome)` method.)
        - ``numeric_agg`` can take on the following five values:
            - `'mean'` : The requested time series are computed by taking the mean for each 
            time interval. E.g. for `plt_type='univariate'`, for each time interval, 
            and for each numeric event feature, the mean of all (transformed) values is 
            is computed. 
            - `'median'` : The requested time series are computed by taking the median for  
            each time interval. E.g. for `plt_type='type_tt'`, for each time interval, 
            and for each numeric event feature, the median of all (transformed) event 
            feature values as well as the median of all TT ratios are computed. 
            - `'min'` : The requested time series are computed by taking the minimum for 
            each time interval. E.g. for `plt_type='type_events_case'`, for each time  
            interval, and for each numeric event feature, both the minimum of all 
            (transformed) values and the minimum of all NEPC ratios are computed. 
            - `'max'` : The requested time series are computed by taking the maximum for 
            each time interval. E.g. for `plt_type='type_outcome'`, for each time  
            interval, and for each numeric event feature, both the maximum of all 
            (transformed) values, the maximum of all (transformed) values for cases 
            with a positive outcome and the maximum of all (transformed) values for 
            cases with a negative outcome are computed. 
            - `'std'` : The requested time series are computed by taking the standard 
            deviation for each time interval. E.g. for `plt_type='type_tt'`, for each  
            time interval, and for each numeric event feature, the standard deviation 
            of all (transformed) event feature values as well as the standard deviation 
            of all TT ratios are computed. 
        - 'numEventFt_transform' determines, for each numerical event feature contained in 
        `numeric_event_list`, how each trace's sequence of numerical event feature values 
        are projected to a single numeric value. It can take on the following eight values: 
            - `'last'` : Each case is assigned the last non-null entry a numerical event feature. 
            - `'first'` : Each case is assigned the first non-null entry of a numerical event feature. 
            - `'mean'` : Each case is assigned the mean value over all its non-null entries of a 
            numerical event feature. 
            - `'median'` : Each case is assigned the median value over all its non-null entries of a 
            numerical event feature. 
            - `'sum'` : Each case is assigned the sum over all its non-null entries of a 
            numerical event feature. 
            - `'prod'` : Each case is assigned the product over all its non-null entries of a 
            numerical event feature. 
            - `'min'` : Each case is assigned the minimum value over all its non-null entries of a 
            numerical event feature. 
            - `'max'` : Each case is assigned the maximum value over all its non-null entries of a 
            numerical event feature. 
        """

        valm._verify_numFt(features = numeric_event_list, ftr_type = 'event', feature_list = self.numerical_eventfeatures,
                            outcome = self.outcome, time_unit = time_unit, frequency = frequency, case_assignment = case_assignment,
                            plt_type = plt_type, numeric_agg = numeric_agg, xtr_outlier_rem = xtr_outlier_rem, 
                            numEventFt_transform = numEventFt_transform)
                            
        up.num_eventfts_evol(log = self.log, numeric_event_list = numeric_event_list, outcome = self.outcome, time_unit = time_unit,
                            frequency = frequency, case_assignment = case_assignment, type = plt_type, numeric_agg = numeric_agg, 
                            xtr_outlier_rem = xtr_outlier_rem, numEventFt_transform = numEventFt_transform)


    def distinct_variants_evol(self, time_unit='days', frequency='weekly', case_assignment = 'first_event', plt_type= 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True, cases_initialized = True):
        """Plots the number of distinct variants over time, as well as the number of 
        distinct previously unseen / distinct new variants over time.

        All cases are grouped into time intervals of which the length is determined by the 
        `frequency` argument. The condition that determines the time interval to which a 
        certain case is assigned, is determined by the `case_assignment` argument. Can be 
        combined with different performance measures by specifying the 
        `plt_type` argument.

        Parameters
        ----------
        time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
            Time unit in which the throughput time of cases is specified, by default `'days'`.
        frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                    '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
            Frequency by which the observations are grouped together, by default `'weekly'`.
        case_assignment : {'first_event', 'last_event', 'max_events'}
            Determines the condition upon which each case is assigned to a certain 
            period, by default `'first_event'`. For a more detailed explanation 
            of the different `case_assignment` options, see Notes.
        plt_type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
            Determines which time series are constructed and visualized, by default `'univariate'`.
            For a more detailed explanation of the different `plt_type` options, see Notes.
        numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
            Determines how periodic quantities are aggregated, by default `'mean'`. The specified
            aggregation function will be applied to all the requested time series, except for 
            those quantities that express fractions or counts (if any). For a more detailed 
            explanation of the different `numeric_agg` options, see Notes.
        xtr_outlier_rem : bool, optional
            If True, the vertical ranges of the plots are only determined by regular  
            values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
            when determining the vertical range, by default `True`.
        cases_initialized : bool, optional
            If True, and `plt_type!='univariate'`, then the method also plots 
            the evolution of the number of cases initialized in each period 
            on the same graph as the one foreseen for either the Throughput Time 
            (`plt_type='type_tt'`), the Number of Events Per Case (`plt_type='type_events_case'`) 
            or the fraction of positive cases (`plt_type='type_outcome'`), by default `True`.

        Notes
        -----
        - ``case_assignment`` can take on the following three values:
            - `'first_event'`: Each case is assigned to the time interval in which 
            its first event occurs. 
            - `'last_event'`: Each case is assigned to the time interval in which 
            its last event occurs
            - `'max_events'`: Out of all the time intervals in which the events of a 
            particular case occur, the case is assigned to the interval in which most 
            of its events occur. Ties among time intervals are broken by assigning the 
            case to the first time interval. 
        - `plt_type` can take one of the following four values:
            - `'univariate'`: Plots the evolution of the amount of distinct variants present in each time period, 
            as well as the amount of distinct new variants introduced in each time period.
            - `'type_tt'`: Next ot the univariate plots, also plots the evolution of the periodically aggregated 
            Throughput Time (TT) (in the time unit specified by the 'time_unit' argument).
            - `'type_events_case'`: Next to the univariate plots, also plots the evolution of the periodically aggregated 
            Number of Events Per Case (NEPC).
            - `'type_outcome'`: Next to the univariate plots, also plots the evolution of the periodically aggregated 
            fraction of cases with outcome = True. (Only applicable if an outcome is already specified upon initialization 
            of the `DynamicLogPlots` instance, or with the `set_outcomeColumn(outcome)` method.)
        """
        valm._verify_distinctvars(outcome = self.outcome, time_unit = time_unit, frequency = frequency, case_assignment = case_assignment, plt_type = plt_type, 
                                    numeric_agg = numeric_agg, xtr_outlier_rem = xtr_outlier_rem, cases_initialized = cases_initialized)
        
        up.distinct_variants_evol(log = self.log, outcome = self.outcome, time_unit = time_unit, frequency = frequency, case_assignment = case_assignment, 
                                    type = plt_type, numeric_agg = numeric_agg, xtr_outlier_rem = xtr_outlier_rem, cases_initialized = cases_initialized)


    def distinct_variants_AdvancedEvol(self, time_unit='days', frequency='weekly', case_assignment = 'first_event', plt_type= 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True, cases_initialized = True):
        """Documentation COMING SOON...

        Parameters
        ----------
        time_unit : str, optional
            _description_, by default `'days'`
        frequency : str, optional
            _description_, by default `'weekly'`
        case_assignment : str, optional
            _description_, by default `'first_event'`
        plt_type : str, optional
            _description_, by default `'univariate'`
        numeric_agg : str, optional
            _description_, by default `'mean'`
        xtr_outlier_rem : bool, optional
            _description_, by default `True`
        cases_initialized : bool, optional
            _description_, by default `True`
        """
        valm._verify_distinctvars(outcome = self.outcome, time_unit = time_unit, frequency = frequency, case_assignment = case_assignment, plt_type = plt_type, 
                                    numeric_agg = numeric_agg, xtr_outlier_rem = xtr_outlier_rem, cases_initialized = cases_initialized)
        
        up.distinct_variants_AdvancedEvol(log = self.log, outcome = self.outcome, time_unit = time_unit, frequency = frequency, case_assignment = case_assignment, 
                                    type = plt_type, numeric_agg = numeric_agg, xtr_outlier_rem = xtr_outlier_rem, cases_initialized = cases_initialized)


    # Utility functions to add or assign new columns to certain attributes: 

    def set_outcomeColumn(self, outcome):
        """Specify an outcome column, or change the outcome column in case 
        one has already been specified.

        Parameters
        ----------
        outcome : str
            Name of the outcome column in the log. (Has to be 
            a binary outcome column with 0 and 1's, and this
            value has to be constant over all events pertaining
            to the same case.)
        """
        valm._verify_addOutCol(self.log, outcome)
        self.outcome = outcome
        

    def add_categorical_caseft(self, case_feature):
        """Specify an additional categorical case feature, on top of the one 
        specified upon initializing your `DynamicLogPlots` instance.

        Parameters
        ----------
        case_feature : str
            Name of the case_feature column in the log. Column has to be of 
            one of the following dtypes: category, object, boolean. 
        """
        if case_feature not in self.categorical_casefeatures:
            valm._verify_addCatCaFt(self.log, case_feature)
            self.categorical_casefeatures.append(case_feature)


    def add_numerical_caseft(self, case_feature):
        """Specify an additional numerical case feature, on top of the one 
        specified upon initializing your `DynamicLogPlots` instance.

        Parameters
        ----------
        case_feature : str
            Name of the case_feature column in the log. Column has to 
            be of a numerical dtype.
        """
        if case_feature not in self.numerical_casefeatures:
            valm._verify_addNumCaFt(self.log, case_feature)
            self.numerical_casefeatures.append(case_feature)

     
    def add_categorical_eventft(self, event_feature):
        """Specify an additional categorical event feature, on top of the one 
        specified upon initializing your `DynamicLogPlots` instance.

        Parameters
        ----------
        event_feature : str
            Name of the event_feature column in the log. Column has to be of 
            one of the following dtypes: category, object, boolean. 
        """
        if event_feature not in self.categorical_eventfeatures:
            valm._verify_addCatEvFt(self.log, event_feature)
            self.categorical_eventfeatures.append(event_feature)


    def add_numerical_eventft(self, event_feature):
        """Specify an additional numerical event feature, on top of the one 
        specified upon initializing your `DynamicLogPlots` instance.

        Parameters
        ----------
        event_feature : str
            Name of the event_feature column in the log. Column has to 
            be of a numerical dtype.
        """
        if event_feature not in self.numerical_eventfeatures:
            valm._verify_addNumEvFt(self.log, event_feature)
            self.numerical_eventfeatures.append(event_feature)
            
    def select_time_range(self, start_date = None, end_date = None):
        """Select only those traces starting after start_date, and ending before end_date.

        Parameters
        ----------
        start_date : str, optional
            Start date string of format "dd/mm/YYYY", by default None
        end_date : str, optional
            End date string of format "dd/mm/YYYY", by default None
        """
        valm._verify_select_time_range(start_date, end_date)
        self.log = select_timerange(log = self.original_log, start_date = start_date, end_date = end_date)
        # Recomputing the ordered list of variants
        case_variant = get_variant_case(self.log)
        self.all_vars = list(case_variant['variant'].value_counts().index)
        self.variant_df = pd.DataFrame(case_variant['variant'].value_counts()).reset_index()
        self.variant_df.columns = ['variant', 'variant count']
        # Recomputing the ordered list of Directly-Follows Relations
        self.dfrelations, self.dfr_df = get_sorted_DFRs(self.log)
    
    # Methods to get dataframes for DFR and Variant encodings in the visuals. 

    def get_DFR_df(self, max_k = None, directly_follows_relations = None, counts = False):
        """Get a `pandas.DataFrame` containing the DFR numbers together 
        with a tuple containing the two corresponding activity labels 
        of each DFR. 

        Parameters
        ----------
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
        dfr_df_filtered : pandas.DataFrame
            Dataframe containing 'DFR number' index and the 'DFR' column 
            containing for each requested DFR the encoded DFR index and 
            the DFR tuple containing the corresponding activity pair 
            respectively. If `counts=True`, the 'DFR count' column that 
            contains the number of occurrences of each DFR over the 
            whole event log is included too. 
        """
        valm._verify_get_DFR_df(max_k, directly_follows_relations, counts, self.dfrelations)
        dfr_df_filtered = get_filtered_dfr_df(self.dfr_df, max_k = max_k, 
                                     directly_follows_relations = directly_follows_relations, 
                                     counts = counts)
        return dfr_df_filtered
    
    def get_var_df(self, max_k = None, variants = None, counts = False):
        """Get a `pandas.DataFrame` containing the variant numbers together 
        with a tuple containing the activity label strings
        of each variant. 

        Parameters
        ----------
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
        valm._verify_get_var_df(max_k, variants, counts, self.all_vars)
        dfr_df_filtered = get_filtered_var_df(self.variant_df, max_k = max_k, 
                                     variants = variants, 
                                     counts = counts)
        return dfr_df_filtered