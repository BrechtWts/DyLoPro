import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from DyLoPro.plotting_utils import determine_time_col, determine_tt_col, get_outcome_percentage,\
                get_maxrange, get_dfr_time, get_variant_case, get_tt_ratios, get_uniq_varcounts,\
                get_newVar_cases, _event_fts_to_tracelvl
from DyLoPro.plot_components import plt_period
from tqdm import tqdm


######################################################################
###   EVOLUTION TOP 'max_k' DIRECTLY-FOLLOWS RELATIONS OVER TIME   ###
######################################################################

def topK_dfr_evol(log, top_k_dfr, outcome = None, time_unit='days', frequency='weekly', case_assignment = 'first_event', type= 'univariate', numeric_agg= 'mean', max_k= 10, xtr_outlier_rem=True):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    top_k_dfr : list of tuple 
        The max_k most frequently occurring Directly-Follows Relations (DFRs).
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome', 'type_dfr_performance'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    max_k : int, optional
        Only the 'max_k' most frequently occurring DFRs are considered, by default 10. 
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.

    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    
    def plt_dfr_uni():
        fig, ax = plt.subplots(max_k+2, 1)
        fig.set_size_inches([20, 6.25*(max_k+2)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case for the {} most frequent Directly-Follows Relations:".format(frequency, 
                    max_k), fontsize=20)

        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e')

        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)


        #   All other plots: 
        for i in range(max_k):
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]

            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k=max_k, 
                    title= "{} evolution of {} #occurrences/case for the {} most common Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k),
                    label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+2], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+2].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')

            if xtr_outlier_rem:
                ax[i+2].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
        if xtr_outlier_rem: 
            ax[1].set_ylim(top = np.nanmax(max_values_percase)*1.05)
            # ax[1].set_ylim(top = global_max_percase*1.05)
        ax[max_k+1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_dfr_tt():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case + evol. of the {} Throughput Time (TT) for the {} most frequent Directly-Follows Relations:".format(frequency, 
                    numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )
        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k): 
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]
            #4
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[0,1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k=max_k, 
                title= "{} evolution of {} #occurrences/case for the {} most common Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k),
                label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+1,0], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+1,0].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')

            plt_period(x, y= period_df[dfr+'_tt'], axes=ax[i+1,1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 1, max_k= 2, 
                title= "{} evolution {} TT for cases with and without DFR {}".format(frequency, numeric_agg, i+1), 
                label = "{} TT ({}) cases in which DFR {} occurs at least once".format(numeric_agg, time_unit, i+1))

            plt_period(x, y= period_df['NOT_'+dfr+'_tt'], axes=ax[i+1,1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 2, max_k= 2, 
                title= "{} evolution {} TT for cases with and without DFR {}".format(frequency, numeric_agg, i+1), 
                label = "{} TT ({}) cases in which DFR {} does NOT occur".format(numeric_agg, time_unit, i+1))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_tt[i], not_max_values_tt[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_percase)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_dfr_events_case(): 
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case + evol. of the {} Number of Events Per Case (NEPC) for the {} most frequent Directly-Follows Relations:".format(frequency, 
                    numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg NEPC
        plt_period(x, period_df['num_events'], ax_0_r, y_label= "{} Number of Events Per Case (NEPC)".format(numeric_agg), label= "{} NEPC".format(numeric_agg),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_numev*1.05)

        #   All other plots: 
        for i in range(max_k):
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]
            #4
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[0,1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k = max_k, 
                    title= "{} evolution of {} #occurrences/case for the {} most common Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k),
                    label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+1,0], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+1,0].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')
            # ax_r.set_ylim([0,1])
            plt_period(x, y= period_df[dfr+'_numev'], axes=ax[i+1,1], y_label = "{} Number Events Per Case (NEPC)".format(numeric_agg), number = 1, max_k = 2, 
                title= "{} evolution {} NEPC for cases with and without DFR {}".format(frequency, numeric_agg, i+1), 
                label = "{} NEPC for cases in which DFR {} occurs at least once".format(numeric_agg, i+1))
            plt_period(x, y= period_df['NOT_'+dfr+'_numev'], axes=ax[i+1,1], y_label = "{} Number Events Per Case (NEPC)".format(numeric_agg), number = 2, max_k = 2, 
                title= "{} evolution {} NEPC for cases with and without DFR {}".format(frequency, numeric_agg, i+1), 
                label = "{} NEPC for cases in which DFR {} does NOT occur".format(numeric_agg, i+1))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_numev[i], not_max_values_numev[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_percase)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_dfr_outcome():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case + evol. of outcome '{}' for the {} most frequent Directly-Follows Relations:".format(frequency, 
                    outcome, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, y= period_df['total'], axes = ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic fraction of cases with outcome = True ( = 1)
        plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            # Accounting for possible outliers in evolution # cases initialized 
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            # Accounting for possible outliers in evolution fraction cases with outcome == True ( == 1)
            ax_0_r.set_ylim(top = max_global_out*1.05)

        #   All other plots: 
        for i in range(max_k):
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]
            #4
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[0,1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k = max_k, 
                    title= "{} evolution of {} #occurrences/case for the {} most common Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k), label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+1,0], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+1,0].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')

            plt_period(x, y= period_df[dfr+'_prc_True'], axes=ax[i+1,1], y_label = "Fraction cases '{}' = True".format(outcome), number = 1, max_k = 2, 
                title= "{} evolution fraction '{}' = True for cases with and without DFR {}".format(frequency, outcome, i+1), 
                label = "Fraction '{}' = True for cases in which DFR {} occurs at least once".format(outcome, i+1))
            plt_period(x, y= period_df['NOT_'+dfr+'_prc_True'], axes=ax[i+1,1], y_label = "Fraction cases '{}' = True".format(outcome), number = 2, max_k = 2, 
                title= "{} evolution fraction {} = True for cases with and without DFR {}".format(frequency, outcome, i+1), 
                label = "Fraction '{}' = True for cases in which DFR {} does NOT occur".format(outcome, i+1))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_out[i], not_max_values_out[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_percase)*1.05)
            

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_dfr_perf():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case + evol. of the {} performance of the {} most frequent Directly-Follows Relations:".format(frequency, 
                    numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k): 
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]
            perf_unit = perf_units_cols[i]
            #4
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[0,1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k=max_k, 
                title= "{} evolution of {} #occurrences/case for the {} most common Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k), label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+1,0], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+1,0].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')

            plt_period(x, y= period_df[dfr+'_perf'], axes=ax[i+1,1], y_label = "Performance ({})".format(perf_unit), 
                title= "{} evolution {} performance (in {}) of DFR {}".format(frequency, numeric_agg, perf_unit, i+1))
            
            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
                # max_y = max(max_values_out[i], not_max_values_out[i])
                # ax[i+1, 1].set_ylim(top = max_y * 1.05)
                ax[i+1, 1].set_ylim(top = max_values_perf[i] * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_percase)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)


    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)

    case_log = log.drop_duplicates(subset='case:concept:name').copy()

    #Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']
    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    # Adding periodic numeric_agg num_events
    if type == 'type_events_case':
        period_numev = case_log.pivot_table(values = 'num_events', index= time_col, aggfunc = numeric_agg, fill_value = 0)
        period_df = period_df.merge(period_numev, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_numev = get_maxrange(period_df[['num_events']])
            max_global_numev = max_global_numev[0]
    
    # Adding periodic fraction of cases with outcome = True (= 1)
    elif type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]


    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]

    # dfg_global, _, _ = pm4py.discover_dfg(log)
    # # Sort the resulting key-value pairs in dictionary dfg_global from high value to low value: 
    # dfg_global = sorted(dfg_global.items(), key=lambda x:x[1], reverse=True)
    # dfg_global = dict(dfg_global)
    # # Get the top k most frequent directly-follows relationships keys (format = tuples):
    # top_k_dfr = list(dfg_global.keys())[:max_k]
    dfr_strings = [str(top_dfr) for top_dfr in top_k_dfr]
    log_loc = log.copy()
    log_loc['next:concept:name'] = log_loc.groupby(['case:concept:name'])['concept:name'].shift(-1)
    log_loc['dfr_start'] = list(zip(log_loc['concept:name'], log_loc['next:concept:name']))
    log_filtered = log_loc[log_loc['dfr_start'].isin(top_k_dfr)][['case:concept:name', 'dfr_start']]
    log_filtered = log_filtered.merge(case_log[['case:concept:name', time_col]], on= 'case:concept:name', how= 'left')

    # Compute periodic counts topk dfrs:
    dfrpercase = log_filtered.pivot_table(values = time_col, index = 'case:concept:name', columns = 'dfr_start', aggfunc = 'count', fill_value = 0)

    # NOTE: underneath you can shorten this with the col naming. One step to many, could also already append the '_percase' here underneath in string_cols 
    string_cols = [str(col) for col in list(dfrpercase.columns)] # To account for different order from dfr_strings
    dfrpercase.columns = string_cols #Strings of the tupples 
    dfrpercase = dfrpercase.merge(case_log[['case:concept:name', time_col]], on = 'case:concept:name', how= 'left')
    #   Periodic numeric_agg of amount of occurrences / case for the given dfrs. 
    period_dfr = dfrpercase.pivot_table(values = string_cols, index = time_col, aggfunc = numeric_agg, fill_value = 0)
    period_dfr.columns = [col+'_percase' for col in list(period_dfr.columns)]
    #       Rearranging order of columns from highest to max_k lowest frequency dfr (for outlier removal):
    period_dfr = period_dfr[[dfr_string+'_percase' for dfr_string in dfr_strings]]
    period_df = period_df.merge(period_dfr, left_index = True, right_index= True, how= 'left')
    if xtr_outlier_rem:
        max_values_percase = get_maxrange(period_dfr)
    

    #Compute periodic fraction of cases with at least one occurrence for the topk dfrs
    period_dfr_cases = log_filtered.pivot_table(values ='case:concept:name', index = time_col, columns = 'dfr_start', aggfunc = pd.Series.nunique, fill_value = 0)
    # string_cols = [str(col)+'_prc' for col in list(period_dfr_cases.columns)]
    period_dfr_cases.columns = [str(col)+'_prc' for col in list(period_dfr_cases.columns)]
    #       Rearranging order of columns from highest to max_k lowest frequency dfr (for outlier removal):
    cols_prc = [dfr_string+'_prc' for dfr_string in dfr_strings]
    period_dfr_cases = period_dfr_cases[cols_prc]
    # Add total and fill Nan's with 0. 
    period_dfr_cases = period_df[['total']].merge(period_dfr_cases, left_index = True, right_index = True, how= 'left').fillna(0)
    period_dfr_cases[cols_prc] = period_dfr_cases[cols_prc].div(period_dfr_cases['total'], axis=0)
    period_dfr_cases = period_dfr_cases.drop(['total'], axis=1).copy()

    period_df = period_df.merge(period_dfr_cases, left_index= True, right_index= True, how= 'left')

    if xtr_outlier_rem: 
        max_values_prc = get_maxrange(period_dfr_cases)
    # global_max_percase = max(max_values_percase)

    x = period_df.index

    if type == 'univariate':
        plt_dfr_uni()
    
    elif type == 'type_tt': 
        # Adding precomputed throughput time column to the log_filtered df:
        log_filt = log_filtered.drop_duplicates(subset= ['case:concept:name', 'dfr_start']).copy()
        log_filt_tt = log_filt.merge(case_log[['case:concept:name', tt_col]], on = 'case:concept:name', how= 'left')
        # Computing periodic numeric_agg of the tt for each of the selected dfrs:
        period_dfr_tt = log_filt_tt.pivot_table(values = tt_col, index = time_col, columns= 'dfr_start', aggfunc = numeric_agg, fill_value = 0)
        period_dfr_tt.columns = [str(col)+'_tt' for col in list(period_dfr_tt.columns)]
        #       Rearranging order of columns from highest to max_k lowest frequency dfr (for outlier removal):
        period_dfr_tt = period_dfr_tt[[dfr_string+'_tt' for dfr_string in dfr_strings]]
        period_df = period_df.merge(period_dfr_tt, left_index = True, right_index= True, how= 'left')
        if xtr_outlier_rem:
            max_values_tt = get_maxrange(period_dfr_tt)

        # Computing these aggregations for cases not containing a certain dfr: 
        dfrpercase_all = log_loc.pivot_table(values = time_col, index = 'case:concept:name', columns = 'dfr_start', aggfunc = 'count', fill_value = 0)
        string_cols = [str(col) for col in list(dfrpercase_all.columns)] # To account for different order from dfr_strings
        dfrpercase_all.columns = string_cols #Strings of the tupples 
        dfrpercase_all = dfrpercase_all[dfr_strings]
        not_max_values_tt = []
        # detected_not_tt = []
        for i in tqdm(range(max_k), desc="Computing additional {} Throughput Time aggregations for each of the {} most frequently occurring DFRs".format(frequency, max_k)):
            dfr_string = dfr_strings[i]
            not_dfr_df = dfrpercase_all[dfrpercase_all[dfr_string] == 0]
            # Exception handling (Possible that a certain dfr occurs in every case):
            num_cases = len(not_dfr_df)
            if num_cases == 0:
                period_df['NOT_'+dfr_string+'_tt'] = [np.nan for _ in range(len(period_df))]
                if xtr_outlier_rem:
                    not_max_values_tt.append(max_values_tt[i])
                # detected_not_tt.append(False)
            else:
                notdfr_caselog = case_log[case_log['case:concept:name'].isin(not_dfr_df.index)]
                notdfr_tt = notdfr_caselog.pivot_table(values = tt_col, index = time_col, aggfunc = numeric_agg, fill_value = 0)
                notdfr_tt.columns = ['NOT_'+dfr_string+'_tt']
                period_df = period_df.merge(notdfr_tt, left_index = True, right_index = True, how= 'left')
                if xtr_outlier_rem:
                    not_max_tt = get_maxrange(notdfr_tt)
                    not_max_values_tt.append(not_max_tt[0])
                    # detected_not_tt.append(not_det_tt[0])

        plt_dfr_tt()

    elif type == 'type_events_case':
        log_filt = log_filtered.drop_duplicates(subset= ['case:concept:name', 'dfr_start']).copy()
        log_filt_numev = log_filt.merge(case_log[['case:concept:name', 'num_events']], on = 'case:concept:name', how= 'left')
        # Computing periodic numeric_agg of the num_events/case for each of the selected dfrs:
        period_dfr_numev = log_filt_numev.pivot_table(values = 'num_events', index = time_col, columns= 'dfr_start', aggfunc = numeric_agg, fill_value = 0)
        period_dfr_numev.columns = [str(col)+'_numev' for col in list(period_dfr_numev.columns)]
        #       Rearranging order of columns from highest to max_k lowest frequency dfr (for outlier removal):
        period_dfr_numev = period_dfr_numev[[dfr_string+'_numev' for dfr_string in dfr_strings]]
        period_df = period_df.merge(period_dfr_numev, left_index = True, right_index= True, how= 'left')
        if xtr_outlier_rem:
            max_values_numev = get_maxrange(period_dfr_numev)


        # Computing these aggregations for cases not containing a certain dfr: 
        dfrpercase_all = log_loc.pivot_table(values = time_col, index = 'case:concept:name', columns = 'dfr_start', aggfunc = 'count', fill_value = 0)
        string_cols = [str(col) for col in list(dfrpercase_all.columns)] # To account for different order from dfr_strings
        dfrpercase_all.columns = string_cols #Strings of the tupples 
        dfrpercase_all = dfrpercase_all[dfr_strings] 
        not_max_values_numev = []
        # detected_not_numev = []
        for i in tqdm(range(max_k), desc="Computing additional {} aggregations of the amount of events per case for each of the {} most frequently occurring DFRs".format(frequency, max_k)):
            dfr_string = dfr_strings[i]
            not_dfr_df = dfrpercase_all[dfrpercase_all[dfr_string] == 0]
            # Exception handling (Possible that a certain dfr occurs in every case):
            num_cases = len(not_dfr_df)
            if num_cases == 0:
                period_df['NOT_'+dfr_string+'_numev'] = [np.nan for _ in range(len(period_df))]
                if xtr_outlier_rem:
                    not_max_values_numev.append(max_values_numev[i])
                # detected_not_numev.append(False)
            else:
                notdfr_caselog = case_log[case_log['case:concept:name'].isin(not_dfr_df.index)]
                notdfr_numev = notdfr_caselog.pivot_table(values = 'num_events', index = time_col, aggfunc = numeric_agg, fill_value = 0)
                notdfr_numev.columns = ['NOT_'+dfr_string+'_numev']
                period_df = period_df.merge(notdfr_numev, left_index = True, right_index = True, how= 'left')
                if xtr_outlier_rem:
                    not_max_numev = get_maxrange(notdfr_numev)
                    not_max_values_numev.append(not_max_numev[0])
                    # detected_not_numev.append(not_det_numev[0])
        plt_dfr_events_case()

    elif type == 'type_outcome':
        # Number of occurrences of each dfr, for each case. 
        dfrpercase_all = log_loc.pivot_table(values = time_col, index = 'case:concept:name', columns = 'dfr_start', aggfunc = 'count', fill_value = 0)
        string_cols = [str(col) for col in list(dfrpercase_all.columns)] # To account for different order from dfr_strings
        dfrpercase_all.columns = string_cols #Strings of the tupples
        # Only retaining information about the max_k most frequent dfr's: 
        dfrpercase_all = dfrpercase_all[dfr_strings] 
        max_values_out = []
        not_max_values_out = []
        for i in tqdm(range(max_k), desc="Computing additional {} outcome aggregations for each of the {} most frequently occurring DFRs".format(frequency, max_k)):
            dfr_string = dfr_strings[i]
            # All cases with at least one occurrence of dfr i
            dfr_df = dfrpercase_all[dfrpercase_all[dfr_string] != 0]
            dfr_caselog = case_log[case_log['case:concept:name'].isin(dfr_df.index)]
            dfr_prcTrue = get_outcome_percentage(filtered_log = dfr_caselog, outcome = outcome, time_col = time_col)
            dfr_prcTrue.columns = [dfr_string+'_prc_True']
            period_df = period_df.merge(dfr_prcTrue, left_index= True, right_index= True, how= 'left')
            if xtr_outlier_rem: 
                max_out = get_maxrange(dfr_prcTrue)
                max_values_out.append(max_out[0])
            # All cases with 0 occurrences of dfr i:
            not_dfr_df = dfrpercase_all[dfrpercase_all[dfr_string] == 0]
            # Exception handling (Possible that a certain dfr occurs in every case):
            num_cases = len(not_dfr_df)
            if num_cases == 0:
                period_df['NOT_'+dfr_string+'_prc_True'] = [np.nan for _ in range(len(period_df))]
                if xtr_outlier_rem:
                    not_max_values_out.append(max_values_out[i])
            else:
                notdfr_caselog = case_log[case_log['case:concept:name'].isin(not_dfr_df.index)]
                not_dfr_prcTrue = get_outcome_percentage(filtered_log = notdfr_caselog, outcome = outcome, time_col = time_col)
                not_dfr_prcTrue.columns = ['NOT_'+dfr_string+'_prc_True']
                period_df = period_df.merge(not_dfr_prcTrue, left_index= True, right_index= True, how= 'left')
                if xtr_outlier_rem:
                    not_max_out = get_maxrange(not_dfr_prcTrue)
                    not_max_values_out.append(not_max_out[0])

        plt_dfr_outcome()
    
    elif type == 'type_dfr_performance':
        period_dfr_perf, perf_units_cols = get_dfr_time(log = log, case_log = case_log,dfr_list = top_k_dfr, time_col = time_col, numeric_agg = numeric_agg)
        period_df = period_df.merge(period_dfr_perf, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem:
            max_values_perf = get_maxrange(period_dfr_perf) 
        plt_dfr_perf()

    plt.show()



#######################################################################
###   EVOLUTION USER-DEFINED DIRECTLY-FOLLOWS RELATIONS OVER TIME   ###
#######################################################################

def dfr_evol(log, directly_follows_relations, outcome = None, time_unit='days', frequency='weekly', case_assignment = 'first_event', type= 'univariate', numeric_agg= 'mean', xtr_outlier_rem=True):
    # warnings.filterwarnings("ignore")
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    directly_follows_relations : list of tuple 
        The DFRs for which the requested time series will be plotted. Each DFR needs to 
        be specified as a tuple that contains 2 strings, referring to the 2 activities in 
        the DFR, e.g. ('activity_a', 'activity_b').
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome', 'type_dfr_performance'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.
    
    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    max_k = len(directly_follows_relations)
    
    def plt_dfr_uni():
        fig, ax = plt.subplots(max_k+2, 1)
        fig.set_size_inches([20, 6.25*(max_k+2)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case for the {} given Directly-Follows Relations:".format(frequency, 
                    max_k), fontsize=20)

        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e')

        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k):
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]

            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k=max_k, 
                    title= "{} evolution of {} #occurrences/case for the {} given Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k),
                    label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+2], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+2].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')
            
            if xtr_outlier_rem:
                ax[i+2].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
        if xtr_outlier_rem: 
            ax[1].set_ylim(top = np.nanmax(max_values_percase)*1.05)

        ax[max_k+1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_dfr_tt():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case + evol. of the {} Throughput Time (TT) for the {} given Directly-Follows Relations:".format(frequency, 
                    numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k): 
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]
            #4
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[0,1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k=max_k, 
                title= "{} evolution of {} #occurrences/case for the {} given Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k),
                label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+1,0], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+1,0].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')

            plt_period(x, y= period_df[dfr+'_tt'], axes=ax[i+1,1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 1, max_k= 2, 
                title= "{} evolution {} TT for cases with and without DFR {}".format(frequency, numeric_agg, i+1), 
                label = "{} TT ({}) cases in which DFR {} occurs at least once".format(numeric_agg, time_unit, i+1))

            plt_period(x, y= period_df['NOT_'+dfr+'_tt'], axes=ax[i+1,1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 2, max_k= 2, 
                title= "{} evolution {} TT for cases with and without DFR {}".format(frequency, numeric_agg, i+1), 
                label = "{} TT ({}) cases in which DFR {} does NOT occur".format(numeric_agg, time_unit, i+1))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_tt[i], not_max_values_tt[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_percase)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_dfr_events_case(): 
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case + evol. of the {} Number of Events Per Case (NEPC) for the {} given Directly-Follows Relations:".format(frequency, 
                    numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg NEPC
        plt_period(x, period_df['num_events'], ax_0_r, y_label= "{} Number of Events Per Case (NEPC)".format(numeric_agg), label= "{} NEPC".format(numeric_agg),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_numev*1.05)

        #   All other plots: 
        for i in range(max_k):
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]
            #4
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[0,1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k = max_k, 
                    title= "{} evolution of {} #occurrences/case for the {} given Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k),
                    label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+1,0], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+1,0].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')
            plt_period(x, y= period_df[dfr+'_numev'], axes=ax[i+1,1], y_label = "{} Number Events Per Case (NEPC)".format(numeric_agg), number = 1, max_k = 2, 
                title= "{} evolution {} NEPC for cases with and without DFR {}".format(frequency, numeric_agg, i+1), 
                label = "{} NEPC for cases in which DFR {} occurs at least once".format(numeric_agg, i+1))
            plt_period(x, y= period_df['NOT_'+dfr+'_numev'], axes=ax[i+1,1], y_label = "{} Number Events Per Case (NEPC)".format(numeric_agg), number = 2, max_k = 2, 
                title= "{} evolution {} NEPC for cases with and without DFR {}".format(frequency, numeric_agg, i+1), 
                label = "{} NEPC for cases in which DFR {} does NOT occur".format(numeric_agg, i+1))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_numev[i], not_max_values_numev[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_percase)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_dfr_outcome():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case + evol. of outcome '{}' for the {} given Directly-Follows Relations:".format(frequency, 
                    outcome, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, y= period_df['total'], axes = ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic fraction of cases with outcome = True ( = 1)
        plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            # Accounting for possible outliers in evolution # cases initialized 
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            # Accounting for possible outliers in evolution fraction cases with outcome == True ( == 1)
            ax_0_r.set_ylim(top = max_global_out*1.05)

        #   All other plots: 
        for i in range(max_k):
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]
            #4
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[0,1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k = max_k, 
                    title= "{} evolution of {} #occurrences/case for the {} given Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k), label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+1,0], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+1,0].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')

            plt_period(x, y= period_df[dfr+'_prc_True'], axes=ax[i+1,1], y_label = "Fraction cases '{}' = True".format(outcome), number = 1, max_k = 2, 
                title= "{} evolution fraction '{}' = True for cases with and without DFR {}".format(frequency, outcome, i+1), 
                label = "Fraction '{}' = True for cases in which DFR {} occurs at least once".format(outcome, i+1))
            plt_period(x, y= period_df['NOT_'+dfr+'_prc_True'], axes=ax[i+1,1], y_label = "Fraction cases '{}' = True".format(outcome), number = 2, max_k = 2, 
                title= "{} evolution fraction {} = True for cases with and without DFR {}".format(frequency, outcome, i+1), 
                label = "Fraction '{}' = True for cases in which DFR {} does NOT occur".format(outcome, i+1))
            
            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_out[i], not_max_values_out[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_percase)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_dfr_perf():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of number directly-follows relations per case + evol. of the {} performance of the {} given Directly-Follows Relations:".format(frequency, 
                    numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k): 
            # dfr= str(top_k_dfr[i])
            dfr = dfr_strings[i]
            perf_unit = perf_units_cols[i]
            #4
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[0,1], y_label = "# Directly-Follows Relations (DFR) per Case", number = i+1, max_k=max_k, 
                title= "{} evolution of {} #occurrences/case for the {} given Directly-Follows Relationships (DFRs)".format(frequency, numeric_agg, max_k), label = "DFR {}".format(i+1))
            # Mean number of dfr occurrences / case
            plt_period(x, y=period_df[dfr+'_percase'], axes=ax[i+1,0], y_label = "Number of DFR {} per case".format(i+1), 
                    label = "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg),
                    title= "DFR {}: {} evolution of {} #occurrences/case".format(i+1, frequency, numeric_agg), location = 'left', color = '#1f77b4')
            ax_r= ax[i+1,0].twinx()
            # Percentage of cases with at least one dfr occurrence
            plt_period(x, y= period_df[dfr+'_prc'], axes=ax_r, y_label = "Fraction cases", 
                    label = "Fraction cases with at least one DFR {}".format(i+1), location = 'right', color = '#ff7f0e')

            plt_period(x, y= period_df[dfr+'_perf'], axes=ax[i+1,1], y_label = "Performance ({})".format(perf_unit), 
                title= "{} evolution {} performance (in {}) of DFR {}".format(frequency, numeric_agg, perf_unit, i+1))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_percase[i]*1.05)
                ax_r.set_ylim(top = max_values_prc[i]*1.05)
                # max_y = max(max_values_out[i], not_max_values_out[i])
                # ax[i+1, 1].set_ylim(top = max_y * 1.05)
                ax[i+1, 1].set_ylim(top = max_values_perf[i] * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_percase)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)


    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)

    case_log = log.drop_duplicates(subset='case:concept:name').copy()

    #Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']
    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    # Adding periodic numeric_agg num_events
    if type == 'type_events_case':
        period_numev = case_log.pivot_table(values = 'num_events', index= time_col, aggfunc = numeric_agg, fill_value = 0)
        period_df = period_df.merge(period_numev, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_numev = get_maxrange(period_df[['num_events']])
            max_global_numev = max_global_numev[0]

    # Adding periodic fraction of cases with outcome = True (= 1)
    elif type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]

    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]

    dfr_strings = [str(top_dfr) for top_dfr in directly_follows_relations]
    log_loc = log.copy()
    log_loc['next:concept:name'] = log_loc.groupby(['case:concept:name'])['concept:name'].shift(-1)
    log_loc['dfr_start'] = list(zip(log_loc['concept:name'], log_loc['next:concept:name']))
    log_filtered = log_loc[log_loc['dfr_start'].isin(directly_follows_relations)][['case:concept:name', 'dfr_start']].copy()
    log_filtered = log_filtered.merge(case_log[['case:concept:name', time_col]], on= 'case:concept:name', how= 'left')

    # Compute periodic counts topk dfrs:
    dfrpercase = log_filtered.pivot_table(values = time_col, index = 'case:concept:name', columns = 'dfr_start', aggfunc = 'count', fill_value = 0)

    # NOTE: underneath you can shorten this with the col naming. One step to many, could also already append the '_percase' here underneath in string_cols 
    string_cols = [str(col) for col in list(dfrpercase.columns)] # To account for different order from dfr_strings
    dfrpercase.columns = string_cols #Strings of the tupples 
    dfrpercase = dfrpercase.merge(case_log[['case:concept:name', time_col]], on = 'case:concept:name', how= 'left')
    #   Periodic numeric_agg of amount of occurrences / case for the given dfrs. 
    period_dfr = dfrpercase.pivot_table(values = string_cols, index = time_col, aggfunc = numeric_agg, fill_value = 0)
    period_dfr.columns = [col+'_percase' for col in list(period_dfr.columns)]
    #       Rearranging order of columns from highest to max_k lowest frequency dfr (for outlier removal):
    period_dfr = period_dfr[[dfr_string+'_percase' for dfr_string in dfr_strings]].copy()
    period_df = period_df.merge(period_dfr, left_index = True, right_index= True, how= 'left')
    if xtr_outlier_rem:
        max_values_percase = get_maxrange(period_dfr)


    #Compute periodic fraction of cases with at least one occurrence for the topk dfrs
    period_dfr_cases = log_filtered.pivot_table(values ='case:concept:name', index = time_col, columns = 'dfr_start', aggfunc = pd.Series.nunique, fill_value = 0)
    # string_cols = [str(col)+'_prc' for col in list(period_dfr_cases.columns)]
    period_dfr_cases.columns = [str(col)+'_prc' for col in list(period_dfr_cases.columns)]
    #       Rearranging order of columns from highest to max_k lowest frequency dfr (for outlier removal):
    cols_prc = [dfr_string+'_prc' for dfr_string in dfr_strings]
    period_dfr_cases = period_dfr_cases[cols_prc]
    # Add total and fill Nan's with 0. 
    period_dfr_cases = period_df[['total']].merge(period_dfr_cases, left_index = True, right_index = True, how= 'left').fillna(0)
    period_dfr_cases[cols_prc] = period_dfr_cases[cols_prc].div(period_dfr_cases['total'], axis=0)
    period_dfr_cases = period_dfr_cases.drop(['total'], axis=1).copy()

    period_df = period_df.merge(period_dfr_cases, left_index= True, right_index= True, how= 'left')

    if xtr_outlier_rem: 
        max_values_prc = get_maxrange(period_dfr_cases)

    x = period_df.index

    if type == 'univariate':
        plt_dfr_uni()

    elif type == 'type_tt': 
        # Adding precomputed throughput time column to the log_filtered df:
        log_filt = log_filtered.drop_duplicates(subset= ['case:concept:name', 'dfr_start']).copy()
        log_filt_tt = log_filt.merge(case_log[['case:concept:name', tt_col]], on = 'case:concept:name', how= 'left')
        # Computing periodic numeric_agg of the tt for each of the selected dfrs:
        period_dfr_tt = log_filt_tt.pivot_table(values = tt_col, index = time_col, columns= 'dfr_start', aggfunc = numeric_agg, fill_value = 0)
        period_dfr_tt.columns = [str(col)+'_tt' for col in list(period_dfr_tt.columns)]
        #       Rearranging order of columns from highest to max_k lowest frequency dfr (for outlier removal):
        period_dfr_tt = period_dfr_tt[[dfr_string+'_tt' for dfr_string in dfr_strings]]
        period_df = period_df.merge(period_dfr_tt, left_index = True, right_index= True, how= 'left')
        if xtr_outlier_rem:
            max_values_tt = get_maxrange(period_dfr_tt)

        # Computing these aggregations for cases not containing a certain dfr: 
        dfrpercase_all = log_loc.pivot_table(values = time_col, index = 'case:concept:name', columns = 'dfr_start', aggfunc = 'count', fill_value = 0)
        string_cols = [str(col) for col in list(dfrpercase_all.columns)] # To account for different order from dfr_strings
        dfrpercase_all.columns = string_cols #Strings of the tupples 
        dfrpercase_all = dfrpercase_all[dfr_strings]
        not_max_values_tt = []
        for i in tqdm(range(max_k), desc="Computing additional {} Throughput Time aggregations for each of the {} given DFRs".format(frequency, max_k)):
            dfr_string = dfr_strings[i]
            not_dfr_df = dfrpercase_all[dfrpercase_all[dfr_string] == 0]
            # Exception handling (Possible that a certain dfr occurs in every case):
            num_cases = len(not_dfr_df)
            if num_cases == 0:
                period_df['NOT_'+dfr_string+'_tt'] = [np.nan for _ in range(len(period_df))]
                if xtr_outlier_rem:
                    not_max_values_tt.append(max_values_tt[i])
            else:
                notdfr_caselog = case_log[case_log['case:concept:name'].isin(not_dfr_df.index)]
                notdfr_tt = notdfr_caselog.pivot_table(values = tt_col, index = time_col, aggfunc = numeric_agg, fill_value = 0)
                notdfr_tt.columns = ['NOT_'+dfr_string+'_tt']
                period_df = period_df.merge(notdfr_tt, left_index = True, right_index = True, how= 'left')
                if xtr_outlier_rem:
                    not_max_tt = get_maxrange(notdfr_tt)
                    not_max_values_tt.append(not_max_tt[0])

        plt_dfr_tt()

    elif type == 'type_events_case':
        log_filt = log_filtered.drop_duplicates(subset= ['case:concept:name', 'dfr_start']).copy()
        log_filt_numev = log_filt.merge(case_log[['case:concept:name', 'num_events']], on = 'case:concept:name', how= 'left')
        # Computing periodic numeric_agg of the num_events/case for each of the selected dfrs:
        period_dfr_numev = log_filt_numev.pivot_table(values = 'num_events', index = time_col, columns= 'dfr_start', aggfunc = numeric_agg, fill_value = 0)
        period_dfr_numev.columns = [str(col)+'_numev' for col in list(period_dfr_numev.columns)]
        #       Rearranging order of columns from highest to max_k lowest frequency dfr (for outlier removal):
        period_dfr_numev = period_dfr_numev[[dfr_string+'_numev' for dfr_string in dfr_strings]]
        period_df = period_df.merge(period_dfr_numev, left_index = True, right_index= True, how= 'left')
        if xtr_outlier_rem:
            max_values_numev = get_maxrange(period_dfr_numev)


        # Computing these aggregations for cases not containing a certain dfr: 
        dfrpercase_all = log_loc.pivot_table(values = time_col, index = 'case:concept:name', columns = 'dfr_start', aggfunc = 'count', fill_value = 0)
        string_cols = [str(col) for col in list(dfrpercase_all.columns)] # To account for different order from dfr_strings
        dfrpercase_all.columns = string_cols #Strings of the tupples 
        dfrpercase_all = dfrpercase_all[dfr_strings] 
        not_max_values_numev = []
        for i in tqdm(range(max_k), desc="Computing additional {} aggregations of the amount of events per case for each of the {} given DFRs".format(frequency, max_k)):
            dfr_string = dfr_strings[i]
            not_dfr_df = dfrpercase_all[dfrpercase_all[dfr_string] == 0]
            # Exception handling (Possible that a certain dfr occurs in every case):
            num_cases = len(not_dfr_df)
            if num_cases == 0:
                period_df['NOT_'+dfr_string+'_numev'] = [np.nan for _ in range(len(period_df))]
                if xtr_outlier_rem:
                    not_max_values_numev.append(max_values_numev[i])
            else:
                notdfr_caselog = case_log[case_log['case:concept:name'].isin(not_dfr_df.index)]
                notdfr_numev = notdfr_caselog.pivot_table(values = 'num_events', index = time_col, aggfunc = numeric_agg, fill_value = 0)
                notdfr_numev.columns = ['NOT_'+dfr_string+'_numev']
                period_df = period_df.merge(notdfr_numev, left_index = True, right_index = True, how= 'left')
                if xtr_outlier_rem:
                    not_max_numev = get_maxrange(notdfr_numev)
                    not_max_values_numev.append(not_max_numev[0])
        plt_dfr_events_case()

    elif type == 'type_outcome':
        # Number of occurrences of each dfr, for each case. 
        dfrpercase_all = log_loc.pivot_table(values = time_col, index = 'case:concept:name', columns = 'dfr_start', aggfunc = 'count', fill_value = 0)
        string_cols = [str(col) for col in list(dfrpercase_all.columns)] # To account for different order from dfr_strings
        dfrpercase_all.columns = string_cols #Strings of the tupples
        # Only retaining information about the max_k given dfr's: 
        dfrpercase_all = dfrpercase_all[dfr_strings] 
        max_values_out = []
        not_max_values_out = []
        for i in tqdm(range(max_k), desc="Computing additional {} outcome aggregations for each of the {} given DFRs".format(frequency, max_k)):
            dfr_string = dfr_strings[i]
            # All cases with at least one occurrence of dfr i
            dfr_df = dfrpercase_all[dfrpercase_all[dfr_string] != 0]
            dfr_caselog = case_log[case_log['case:concept:name'].isin(dfr_df.index)]
            dfr_prcTrue = get_outcome_percentage(filtered_log = dfr_caselog, outcome = outcome, time_col = time_col)
            dfr_prcTrue.columns = [dfr_string+'_prc_True']
            period_df = period_df.merge(dfr_prcTrue, left_index= True, right_index= True, how= 'left')
            if xtr_outlier_rem: 
                max_out = get_maxrange(dfr_prcTrue)
                max_values_out.append(max_out[0])
            # All cases with 0 occurrences of dfr i:
            not_dfr_df = dfrpercase_all[dfrpercase_all[dfr_string] == 0]
            # Exception handling (Possible that a certain dfr occurs in every case):
            num_cases = len(not_dfr_df)
            if num_cases == 0:
                period_df['NOT_'+dfr_string+'_prc_True'] = [np.nan for _ in range(len(period_df))]
                if xtr_outlier_rem:
                    not_max_values_out.append(max_values_out[i])
            else:
                notdfr_caselog = case_log[case_log['case:concept:name'].isin(not_dfr_df.index)]
                not_dfr_prcTrue = get_outcome_percentage(filtered_log = notdfr_caselog, outcome = outcome, time_col = time_col)
                not_dfr_prcTrue.columns = ['NOT_'+dfr_string+'_prc_True']
                period_df = period_df.merge(not_dfr_prcTrue, left_index= True, right_index= True, how= 'left')
                if xtr_outlier_rem:
                    not_max_out = get_maxrange(not_dfr_prcTrue)
                    not_max_values_out.append(not_max_out[0])

        plt_dfr_outcome()

    
    elif type == 'type_dfr_performance':
        period_dfr_perf, perf_units_cols = get_dfr_time(log = log, case_log = case_log, dfr_list = directly_follows_relations, time_col = time_col, numeric_agg = numeric_agg)
        period_df = period_df.merge(period_dfr_perf, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem:
            max_values_perf = get_maxrange(period_dfr_perf) 
        plt_dfr_perf()

    plt.show()


######################################################################
###             EVOLUTION TOP 'max_k' VARIANTS OVER TIME           ###
######################################################################

def topK_variants_evol(log, top_k_vars, outcome = None, time_unit='days', frequency='weekly', case_assignment = 'first_event', type= 'univariate', numeric_agg= 'mean', max_k= 10, xtr_outlier_rem = True):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    top_k_vars : list of tuple 
        The 'max_k' most frequently occurring variants.
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_outcome'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    max_k : int, optional
        Only the 'max_k' most frequently occurring variants are considered, by default 10. 
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.

    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    # Inner functions belonging to a specific plot type:
    def plt_vars_uni():
        fig, ax = plt.subplots(max_k+2, 1)
        fig.set_size_inches([20, 6.25*(max_k+2)])
        st = plt.suptitle("{} evolution of fraction cases belonging to a variant for the {} most frequent variants:".format(frequency, 
                    max_k), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )
        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)


        #   All other plots: 
        for i in range(max_k):
            #4
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} evolution fraction of initialized cases belonging to the {} most common variants".format(frequency, max_k), label = "Variant {}".format(i+1))
            #3
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[i+2], y_label = "Variant {}: fraction cases".format(i+1), 
                    title= "Variant {}: {} evolution of fraction cases it accounts for".format(i+1, frequency))
            if xtr_outlier_rem:
                ax[i+2].set_ylim(top = max_values_prc[i]*1.05)
        if xtr_outlier_rem:
            ax[1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k+1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_vars_tt():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of fraction cases belonging to a variant for the {} most frequent variants + evolution {} Throughput Time :".format(frequency, 
                    max_k, numeric_agg), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e')

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k):
            #4
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[0,1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                title = "{} evolution fraction of initialized cases belonging to the {} most common variants".format(frequency, max_k), label = "Variant {}".format(i+1))
            #3
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[i+1,0], y_label = "Variant {}: fraction cases".format(i+1), 
                title= "Variant {}: {} evolution of fraction cases it accounts for".format(i+1, frequency))

            plt_period(x, y= period_df["Variant {}_tt".format(i+1)], axes=ax[i+1,1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 1, max_k=2, 
                title = "{} evolution {} Throughput Time (TT) Variant {}".format(frequency, numeric_agg, i+1), label = "{} TT ({}) for cases of Variant {}".format(numeric_agg, time_unit, i+1))
            plt_period(x, y= period_df["NOT_Variant {}_tt".format(i+1)], axes=ax[i+1,1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 2, max_k=2, 
                title = "{} evolution {} Throughput Time (TT) Variant {}".format(frequency, numeric_agg, i+1), label = "{} TT ({}) for cases NOT of Variant {}".format(numeric_agg, time_unit, i+1))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_tt[i], not_max_values_tt[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_vars_outcome():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of fraction cases belonging to a variant for the {} most frequent variants + evolution of fraction cases with '{}' = True:".format(frequency, 
                    max_k, outcome), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic fraction of cases with outcome = True ( = 1)
        plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            # Accounting for possible outliers in evolution # cases initialized 
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            # Accounting for possible outliers in evolution fraction cases with outcome == True ( == 1)
            ax_0_r.set_ylim(top = max_global_out*1.05)

        #   All other plots: 
        for i in range(max_k):
            #4
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[0,1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                title = "{} evolution fraction of initialized cases belonging to the {} most common variants".format(frequency, max_k), label = "Variant {}".format(i+1))
            #3
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[i+1,0], y_label = "Variant {}: fraction cases".format(i+1), 
                title= "Variant {}: {} evolution of fraction cases it accounts for".format(i+1, frequency))

            plt_period(x, y= period_df["Variant {}_prc_True".format(i+1)], axes=ax[i+1,1], y_label = "Fraction cases '{}' = True".format(outcome), number = 1, max_k=2, 
                title = "{} evolution fraction '{}' = True for cases of Variant {}".format(frequency, outcome, i+1),
                label = "Fraction '{}' = True for cases of Variant {}".format(outcome, i+1))
            plt_period(x, y= period_df["NOT_Variant {}_prc_True".format(i+1)], axes=ax[i+1,1], y_label = "Fraction cases '{}' = True".format(outcome), number = 2, max_k=2, 
                title = "{} evolution fraction '{}' = True for cases of Variant {}".format(frequency, outcome, i+1),
                label = "Fraction '{}' = True for cases of Variant {}".format(outcome, i+1))

            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'fraction evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                # Accounting for possible outliers in right 'Fraction cases with outcome == 1 evolution plot'
                max_y = max(max_values_out[i], not_max_values_out[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)

        if xtr_outlier_rem:
            # Accounting for possible outliers in global 'fraction evolution plot'
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)
        
        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

            
    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)

    # Get dataframe containing the case id and variant (as a tuple of activity strings) for each case:
    case_variant = get_variant_case(log)
    
    variant_id = [i for i in range(1, max_k+1)]

    case_log = log.drop_duplicates(subset='case:concept:name').copy()
    #Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']
    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    
    # Adding periodic fraction of cases with outcome = True (= 1)
    if type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]


    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]
    
    # Computing aggregations for the given variants: 
    if type == 'type_outcome':
        local_log = case_log[['case:concept:name', time_col, tt_col, outcome]]
    else:
        local_log = case_log[['case:concept:name', time_col, tt_col]]
    local_log = local_log.merge(case_variant, on = 'case:concept:name', how= 'left')
    local_log_sliced = local_log[local_log['variant'].isin(top_k_vars)]

    # Computing periodic fractions of cases represented by each of the given variants:
    var_counts = local_log_sliced.pivot_table(values = 'case:concept:name', index = time_col, columns = 'variant', aggfunc = 'count', fill_value = 0)
    # Re-order from most frequent to max_k'th most frequent variant
    var_counts = var_counts[top_k_vars]
    # Rename columns for efficiency: 
    string_cols = ["Variant {}_prc".format(i) for i in range(1, max_k+1)]
    var_counts.columns = string_cols
    var_counts = period_df[['total']].merge(var_counts, left_index = True, right_index = True, how= 'left').fillna(0)
    var_counts[string_cols] = var_counts[string_cols].div(var_counts['total'], axis=0)
    var_counts = var_counts.drop(['total'], axis=1).copy()
    period_df = period_df.merge(var_counts, left_index= True, right_index= True, how='left')
    if xtr_outlier_rem:
        max_values_prc = get_maxrange(var_counts)
    
    x = period_df.index

    if type == 'univariate':
        plt_vars_uni()

    elif type == 'type_tt':
        period_var_tt = local_log_sliced.pivot_table(values = tt_col , index = time_col, columns = 'variant', aggfunc = numeric_agg, fill_value = 0)
        # Re-order from most frequent to max_k'th most frequent variant
        period_var_tt = period_var_tt[top_k_vars]
        # Rename columns for efficiency: 
        string_cols = ["Variant {}_tt".format(i) for i in range(1, max_k+1)]
        period_var_tt.columns = string_cols
        period_df = period_df.merge(period_var_tt, left_index= True, right_index= True, how = 'left')

        if xtr_outlier_rem:
            max_values_tt= get_maxrange(period_var_tt)
        
        not_max_values_tt = []
        for i in tqdm(range(max_k), desc="Computing additional {} Throughput Time aggregations for each of the {} most frequently occurring variants".format(frequency, max_k)):
            vart = top_k_vars[i]
            not_var_log = local_log[local_log['variant'] != vart]
            not_var_tt = not_var_log.pivot_table(values = tt_col, index = time_col, aggfunc = numeric_agg, fill_value = 0)
            not_var_tt.columns = ['NOT_Variant {}_tt'.format(i+1)]
            period_df = period_df.merge(not_var_tt, left_index= True, right_index= True, how = 'left')

            if xtr_outlier_rem:
                not_max_tt= get_maxrange(not_var_tt)
                not_max_values_tt.append(not_max_tt[0])

        plt_vars_tt()

    elif type == 'type_outcome':
        max_values_out = []
        not_max_values_out = []
        for i in tqdm(range(max_k), desc="Computing additional {} outcome aggregations for each of the {} most frequently occurring variants".format(frequency, max_k)):
            vart = top_k_vars[i]
            var_log = local_log_sliced[local_log_sliced['variant'] == vart]
            var_log_prcTrue = get_outcome_percentage(filtered_log = var_log, outcome = outcome, time_col = time_col)
            var_log_prcTrue.columns = ["Variant {}_prc_True".format(i+1)]
            period_df = period_df.merge(var_log_prcTrue, left_index = True, right_index = True, how = 'left')

            not_var_log = local_log[local_log['variant'] != vart]
            not_var_log_prcTrue = get_outcome_percentage(filtered_log = not_var_log, outcome = outcome, time_col = time_col)
            not_var_log_prcTrue.columns = ["NOT_Variant {}_prc_True".format(i+1)]
            period_df = period_df.merge(not_var_log_prcTrue, left_index = True, right_index = True, how = 'left')

            if xtr_outlier_rem:
                max_out = get_maxrange(var_log_prcTrue)
                max_values_out.append(max_out[0])
                not_max_out = get_maxrange(not_var_log_prcTrue)
                not_max_values_out.append(not_max_out[0])

        plt_vars_outcome()

    plt.show()



######################################################################
###           EVOLUTION USER-DEFINED VARIANTS OVER TIME            ###
######################################################################


def variants_evol(log, variants, outcome = None, time_unit='days', frequency='weekly', case_assignment = 'first_event', type= 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    variants : list of tuple
        The variants for which the requested time series will be plotted. Each variant needs to 
        be specified as a tuple that contains N strings, referring to the N activities that 
        constitute that variant. 
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_outcome'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.

    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    max_k = len(variants)
    # Inner functions belonging to a specific plot type:
    def plt_vars_uni():
        fig, ax = plt.subplots(max_k+2, 1)
        fig.set_size_inches([20, 6.25*(max_k+2)])
        st = plt.suptitle("{} evolution of fraction cases belonging to a variant for the {} given variants:".format(frequency, 
                    max_k), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )
        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)


        #   All other plots: 
        for i in range(max_k):
            #4
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} evolution fraction of initialized cases belonging to the {} given variants".format(frequency, max_k), label = "Variant {}".format(i+1))
            #3
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[i+2], y_label = "Variant {}: fraction cases".format(i+1), 
                    title= "Variant {}: {} evolution of fraction cases it accounts for".format(i+1, frequency))

            if xtr_outlier_rem:
                ax[i+2].set_ylim(top = max_values_prc[i]*1.05)
        if xtr_outlier_rem:
            ax[1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k+1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_vars_tt():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of fraction cases belonging to a variant for the {} given variants + evolution {} Throughput Time :".format(frequency, 
                    max_k, numeric_agg), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e')

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k):
            #4
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[0,1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                title = "{} evolution fraction of initialized cases belonging to the {} given variants".format(frequency, max_k), label = "Variant {}".format(i+1))
            #3
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[i+1,0], y_label = "Variant {}: fraction cases".format(i+1), 
                title= "Variant {}: {} evolution of fraction cases it accounts for".format(i+1, frequency))

            plt_period(x, y= period_df["Variant {}_tt".format(i+1)], axes=ax[i+1,1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 1, max_k=2, 
                title = "{} evolution {} Throughput Time (TT) Variant {}".format(frequency, numeric_agg, i+1), label = "{} TT ({}) for cases of Variant {}".format(numeric_agg, time_unit, i+1))
            plt_period(x, y= period_df["NOT_Variant {}_tt".format(i+1)], axes=ax[i+1,1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 2, max_k=2, 
                title = "{} evolution {} Throughput Time (TT) Variant {}".format(frequency, numeric_agg, i+1), label = "{} TT ({}) for cases NOT of Variant {}".format(numeric_agg, time_unit, i+1))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_tt[i], not_max_values_tt[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_vars_outcome():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{} evolution of fraction cases belonging to a variant for the {} given variants + evolution of fraction cases with '{}' = True:".format(frequency, 
                    max_k, outcome), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic fraction of cases with outcome = True ( = 1)
        plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                location = 'right', color= '#ff7f0e' )
        
        if xtr_outlier_rem:
            # Accounting for possible outliers in evolution # cases initialized 
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            # Accounting for possible outliers in evolution fraction cases with outcome == True ( == 1)
            ax_0_r.set_ylim(top = max_global_out*1.05)

        #   All other plots: 
        for i in range(max_k):
            #4
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[0,1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                title = "{} evolution fraction of initialized cases belonging to the {} given variants".format(frequency, max_k), label = "Variant {}".format(i+1))
            #3
            plt_period(x, y=period_df["Variant {}_prc".format(i+1)], axes=ax[i+1,0], y_label = "Variant {}: fraction cases".format(i+1), 
                title= "Variant {}: {} evolution of fraction cases it accounts for".format(i+1, frequency))

            plt_period(x, y= period_df["Variant {}_prc_True".format(i+1)], axes=ax[i+1,1], y_label = "Fraction cases '{}' = True".format(outcome), number = 1, max_k=2, 
                title = "{} evolution fraction '{}' = True for cases of Variant {}".format(frequency, outcome, i+1),
                label = "Fraction '{}' = True for cases of Variant {}".format(outcome, i+1))
            plt_period(x, y= period_df["NOT_Variant {}_prc_True".format(i+1)], axes=ax[i+1,1], y_label = "Fraction cases '{}' = True".format(outcome), number = 2, max_k=2, 
                title = "{} evolution fraction '{}' = True for cases of Variant {}".format(frequency, outcome, i+1),
                label = "Fraction '{}' = True for cases of Variant {}".format(outcome, i+1))

            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'fraction evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                # Accounting for possible outliers in right 'Fraction cases with outcome == 1 evolution plot'
                max_y = max(max_values_out[i], not_max_values_out[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            # Accounting for possible outliers in global 'fraction evolution plot'
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
            
    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)

    # Get dataframe containing the case id and variant (as a tuple of activity strings) for each case:
    case_variant = get_variant_case(log)

    variant_id = [i for i in range(1, max_k+1)]
    # idx_var_mapping = dict(zip(variant_id, variants))

    case_log = log.drop_duplicates(subset='case:concept:name').copy()
    #Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']
    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    # Adding periodic fraction of cases with outcome = True (= 1)
    if type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]


    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]
    
    # Computing aggregations for the given variants: 
    if type == 'type_outcome':
        local_log = case_log[['case:concept:name', time_col, tt_col, outcome]]
    else:
        local_log = case_log[['case:concept:name', time_col, tt_col]]
    local_log = local_log.merge(case_variant, on = 'case:concept:name', how= 'left')
    local_log_sliced = local_log[local_log['variant'].isin(variants)]

    # Computing periodic fractions of cases represented by each of the given variants:
    var_counts = local_log_sliced.pivot_table(values = 'case:concept:name', index = time_col, columns = 'variant', aggfunc = 'count', fill_value = 0)
    # Re-order according to the order specified in variants.
    var_counts = var_counts[variants]
    # Rename columns for efficiency: 
    string_cols = ["Variant {}_prc".format(i) for i in range(1, max_k+1)]
    var_counts.columns = string_cols
    var_counts = period_df[['total']].merge(var_counts, left_index = True, right_index = True, how= 'left').fillna(0)
    var_counts[string_cols] = var_counts[string_cols].div(var_counts['total'], axis=0)
    var_counts = var_counts.drop(['total'], axis=1).copy()
    period_df = period_df.merge(var_counts, left_index= True, right_index= True, how='left')

    if xtr_outlier_rem:
        max_values_prc = get_maxrange(var_counts)
    
    x = period_df.index

    if type == 'univariate':
        plt_vars_uni()

    elif type == 'type_tt':
        period_var_tt = local_log_sliced.pivot_table(values = tt_col , index = time_col, columns = 'variant', aggfunc = numeric_agg, fill_value = 0)
        # Re-order according to the order specified in variants.
        period_var_tt = period_var_tt[variants]
        # Rename columns for efficiency: 
        string_cols = ["Variant {}_tt".format(i) for i in range(1, max_k+1)]
        period_var_tt.columns = string_cols
        period_df = period_df.merge(period_var_tt, left_index= True, right_index= True, how = 'left')
        
        if xtr_outlier_rem:
            max_values_tt= get_maxrange(period_var_tt)
        
        not_max_values_tt = []
        for i in tqdm(range(max_k), desc="Computing additional {} Throughput Time aggregations for each of the {} given variants".format(frequency, max_k)):
            vart = variants[i]
            not_var_log = local_log[local_log['variant'] != vart]
            not_var_tt = not_var_log.pivot_table(values = tt_col, index = time_col, aggfunc = numeric_agg, fill_value = 0)
            not_var_tt.columns = ['NOT_Variant {}_tt'.format(i+1)]
            period_df = period_df.merge(not_var_tt, left_index= True, right_index= True, how = 'left')

            if xtr_outlier_rem:
                not_max_tt = get_maxrange(not_var_tt)
                not_max_values_tt.append(not_max_tt[0])

        plt_vars_tt()

    elif type == 'type_outcome':
        max_values_out = []
        not_max_values_out = []
        for i in tqdm(range(max_k), desc="Computing additional {} outcome aggregations for each of the {} given variants".format(frequency, max_k)):
            vart = variants[i]
            var_log = local_log_sliced[local_log_sliced['variant'] == vart]
            var_log_prcTrue = get_outcome_percentage(filtered_log = var_log, outcome = outcome, time_col = time_col)
            var_log_prcTrue.columns = ["Variant {}_prc_True".format(i+1)]
            period_df = period_df.merge(var_log_prcTrue, left_index = True, right_index = True, how = 'left')

            not_var_log = local_log[local_log['variant'] != vart]
            not_var_log_prcTrue = get_outcome_percentage(filtered_log = not_var_log, outcome = outcome, time_col = time_col)
            not_var_log_prcTrue.columns = ["NOT_Variant {}_prc_True".format(i+1)]
            period_df = period_df.merge(not_var_log_prcTrue, left_index = True, right_index = True, how = 'left')

            if xtr_outlier_rem:
                max_out = get_maxrange(var_log_prcTrue)
                max_values_out.append(max_out[0])
                not_max_out = get_maxrange(not_var_log_prcTrue)
                not_max_values_out.append(not_max_out[0])
        
        plt_vars_outcome()
        
    print(pd.DataFrame(list(zip(variant_id, variants)), columns=['Variant ID', 'Activity Sequence']).set_index('Variant ID').to_string())
    plt.show()


#############################################################################################################################
#############################################################################################################################
###############                                                                                               ###############
###############                             EVOLUTION CASE FEATURES OVER TIME                                 ###############
###############                                                                                               ###############
#############################################################################################################################
#############################################################################################################################



######################################################################
###         EVOLUTION CATEGORICAL CASE FEATURES OVER TIME          ###
######################################################################

def topK_categorical_caseftr_evol(log, case_feature, outcome = None, time_unit = 'days', frequency = 'weekly', case_assignment = 'first_event', type = 'univariate', numeric_agg = 'mean', max_k = 10, xtr_outlier_rem = True):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    case_feature : str
        Column name of the categorical case feature.
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    max_k : int, optional
        Only the 'max_k' most frequently occurring levels of the feature are considered, by default 10. 
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.

    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    # Inner functions belonging to a specific plot type:
    def plt_caseft_uni():
        fig, ax = plt.subplots(max_k+2, 1)
        fig.set_size_inches([20, 6.25*(max_k+2)])
        st = plt.suptitle("{}: {} evolution fractions for each of the {} most frequent levels".format(case_feature, frequency, max_k), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )


        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)


        #   All other plots: 
        for i in range(max_k):
            level = levels[i]
            #4
            plt_period(x, y = period_df[level+'_prc'], axes = ax[1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} fraction of cases belonging to the {} most common levels of {}".format(frequency, max_k, case_feature), label = '{}. {}'.format(i+1, level))
            #3
            plt_period(x, y=period_df[level+'_prc'], axes=ax[i+2], y_label = level+": fraction cases", 
                    title= "{}. '{}' = '{}': {} evolution fraction cases".format(i+1, case_feature, level, frequency))

            if xtr_outlier_rem:
                ax[i+2].set_ylim(top = max_values_prc[i]*1.05)
        if xtr_outlier_rem:
            ax[1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

            
        ax[max_k+1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)


    def plt_caseft_tt():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{}: {} evolution fractions and {} Throughput Time (TT) for each of the {} most frequent levels".format(case_feature, frequency,
                        numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k):
            level = levels[i]
            #4
            plt_period(x, y=period_df[level+'_prc'], axes = ax[0, 1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} fraction of cases belonging to the {} most common levels of {}".format(frequency, max_k, case_feature), label = '{}. {}'.format(i+1, level))
            #3
            plt_period(x, y=period_df[level+'_prc'], axes=ax[i+1, 0], y_label = level+": fraction cases", 
                    title= "{}. '{}' = '{}': {} evolution fraction cases".format(i+1, case_feature, level, frequency))
            #4
            plt_period(x, y= period_df[level+"_tt"], axes = ax[i+1, 1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 1, max_k=2, 
                    title = "{}. '{}' = '{}': {} evolution {} TT".format(i+1, case_feature, level, frequency, numeric_agg),
                    label = "{} TT ({}) for cases with '{}' = '{}'".format(numeric_agg, time_unit, case_feature, level))
            plt_period(x, y= period_df["NOT_"+level+"_tt"], axes = ax[i+1, 1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 2, max_k=2, 
                    title = "{}. '{}' = '{}': {} evolution {} TT".format(i+1, case_feature, level, frequency, numeric_agg),
                    label = "{} TT ({}) for cases with '{}' NOT = '{}'".format(numeric_agg, time_unit, case_feature, level))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_tt[i], not_max_values_tt[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_caseft_events_case(): 
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{}: {} evolution fractions and {} Number of Events Per Case (NEPC) for each of the {} most frequent levels".format(case_feature, frequency,
                        numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of NEPC
        plt_period(x, period_df['num_events'], ax_0_r, y_label= "{} Number of Events Per Case (NEPC)".format(numeric_agg), label= "{} NEPC".format(numeric_agg),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_numev*1.05)

        #   All other plots: 
        for i in range(max_k):
            level = levels[i]
            #4
            plt_period(x, y=period_df[level+'_prc'], axes = ax[0, 1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} fraction of cases belonging to the {} most common levels of {}".format(frequency, max_k, case_feature), label = '{}. {}'.format(i+1, level))
            #3
            plt_period(x, y=period_df[level+'_prc'], axes=ax[i+1, 0], y_label = level+": fraction cases", 
                    title= "{}. '{}' = '{}': {} evolution fraction cases".format(i+1, case_feature, level, frequency))

            #4
            plt_period(x, y= period_df[level+"_numev"], axes = ax[i+1, 1], y_label = "{} Number Events Per Case (NEPC)".format(numeric_agg), number = 1, max_k=2, 
                    title = "{}. '{}' = '{}': {} evolution {} NEPC".format(i+1, case_feature, level, frequency, numeric_agg),
                    label = "{} NEPC for cases with '{}' = '{}'".format(numeric_agg, case_feature, level))
            plt_period(x, y= period_df["NOT_"+level+"_numev"], axes = ax[i+1, 1], y_label = "{} Number Events Per Case (NEPC)".format(numeric_agg), number = 2, max_k=2, 
                    title = "{}. '{}' = '{}': {} evolution {} NEPC".format(i+1, case_feature, level, frequency, numeric_agg),
                    label = "{} NEPC for cases with '{}' NOT = '{}'".format(numeric_agg, case_feature, level))

            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'fraction evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                # Accounting for possible outliers in right 'NEPC evolution plot'
                max_y = max(max_values_numev[i], not_max_values_numev[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            # Accounting for possible outliers in global 'fraction evolution plot'
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_caseft_outcome():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{}: {} evolution fractions and fraction of cases with '{}' = True for each of the {} most frequent levels".format(case_feature, frequency,
                        outcome, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic fraction of cases with outcome = True ( = 1)
        plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            # Accounting for possible outliers in evolution # cases initialized 
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            # Accounting for possible outliers in evolution fraction cases with outcome == True ( == 1)
            ax_0_r.set_ylim(top = max_global_out*1.05)

        #   All other plots: 
        for i in range(max_k):
            level = levels[i]
            #4
            plt_period(x, y=period_df[level+'_prc'], axes = ax[0, 1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} fraction of cases belonging to the {} most common levels of {}".format(frequency, max_k, case_feature), label = '{}. {}'.format(i+1, level))
            #3
            plt_period(x, y=period_df[level+'_prc'], axes=ax[i+1, 0], y_label = level+": fraction cases", 
                    title= "{}. '{}' = '{}': {} evolution fraction cases".format(i+1, case_feature, level, frequency))

            #4
            plt_period(x, y= period_df[level+'_prc_True'], axes = ax[i+1, 1], y_label = "Fraction cases '{}' = True".format(outcome), number = 1, max_k=2, 
                    title = "{}. '{}' = '{}': {} evolution fraction '{}' = True".format(i+1, case_feature, level, frequency, outcome),
                    label = "Fraction '{}' = True for cases with '{}' = '{}'".format(outcome, case_feature, level))
            plt_period(x, y= period_df['NOT_'+level+'_prc_True'], axes = ax[i+1, 1], y_label = "Fraction cases '{}' = True".format(outcome), number = 2, max_k=2, 
                    title = "{}. '{}' = '{}': {} evolution fraction '{}' = True".format(i+1, case_feature, level, frequency, outcome),
                    label = "Fraction '{}' = True for cases with '{}' NOT = '{}'".format(outcome, case_feature, level))

            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'fraction evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                # Accounting for possible outliers in right 'Fraction cases with outcome == 1 evolution plot'
                max_y = max(max_values_out[i], not_max_values_out[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            # Accounting for possible outliers in global 'fraction evolution plot'
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)


    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)

    # Accounting for boolean categoricals 
    log[case_feature] = log[case_feature].astype('str')
    case_log = log.drop_duplicates(subset = 'case:concept:name').copy()
    # Removing possible missing values
    case_log = case_log.dropna(subset = case_feature).copy()

    levels = list(case_log[case_feature].value_counts().index) #sorted
    num_levels = len(levels)
    if num_levels == 1:
        print("Case feature {} only contains 1 level, and hence does not carry any useful information.".format(case_feature))
        return
    if max_k < num_levels: 
        levels = levels[:max_k]
        # max_k = num_levels
    elif max_k > num_levels:
        max_k = num_levels
        levels = levels[:max_k]
    
    # Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']
    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    # Adding periodic numeric_agg num_events
    if type == 'type_events_case':
        period_numev = case_log.pivot_table(values = 'num_events', index= time_col, aggfunc = numeric_agg, fill_value = 0)
        period_df = period_df.merge(period_numev, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_numev = get_maxrange(period_df[['num_events']])
            max_global_numev = max_global_numev[0]
    
    # Adding periodic fraction of cases with outcome = True (= 1)
    elif type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]

    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values = tt_col, index = time_col, aggfunc = numeric_agg, fill_value = 0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]

    x = period_df.index
    # Computing the needed values for the max_k most frequent levels:
    case_sliced = case_log[case_log[case_feature].isin(levels)]

    case_df = case_sliced.pivot_table(values= 'case:concept:name', index = time_col, columns = case_feature, aggfunc = 'count', fill_value = 0)
    # Reorder from highest frequency level to lowest frequency level
    case_df = case_df[levels]
    # Rename columns
    string_cols = [level+'_prc' for level in levels]
    case_df.columns = string_cols
    # Add total and fill Nan's with 0. 
    case_df = period_df[['total']].merge(case_df, left_index = True, right_index = True, how= 'left').fillna(0)
    case_df[string_cols] = case_df[string_cols].div(case_df['total'], axis=0)
    case_df = case_df.drop(['total'], axis=1).copy()

    period_df = period_df.merge(case_df, left_index = True, right_index = True, how = 'left')

    if xtr_outlier_rem:
        max_values_prc = get_maxrange(case_df)

    
    if type == 'univariate':
        plt_caseft_uni()
    elif type == 'type_tt':
        case_tt = case_sliced.pivot_table(values= tt_col, index = time_col,columns = case_feature, aggfunc = numeric_agg, fill_value = 0)
        # Reorder from highest frequency level to lowest frequency level
        case_tt = case_tt[levels]
        # Rename columns
        case_tt.columns = [col+'_tt' for col in list(case_tt.columns)]
        period_df = period_df.merge(case_tt, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem:
            # max_values_tt, detected_tt = get_maxrange(case_tt)
            max_values_tt = get_maxrange(case_tt)

        not_max_values_tt = []
        # detected_not_tt = []
        for level in tqdm(levels, desc = "Computing additional {} aggregations of the Throughput Time for each of the {} most frequently occurring levels".format(frequency, max_k)):
            not_case_tt = case_log[case_log[case_feature]!=level].pivot_table(values = tt_col, index = time_col, aggfunc = numeric_agg, fill_value = 0)
            not_case_tt.columns = ['NOT_'+level+'_tt']
            period_df = period_df.merge(not_case_tt, left_index = True, right_index= True, how = 'left')
            if xtr_outlier_rem:
                not_max_tt = get_maxrange(not_case_tt)
                not_max_values_tt.append(not_max_tt[0])

        plt_caseft_tt()

    elif type == 'type_events_case': 
        case_numev = case_sliced.pivot_table(values= 'num_events', index = time_col, columns = case_feature, aggfunc = numeric_agg, fill_value = 0)
        # Reorder from highest frequency level to lowest frequency level
        case_numev = case_numev[levels]
        # Rename columns
        case_numev.columns = [col+'_numev' for col in list(case_numev.columns)]
        period_df = period_df.merge(case_numev, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem:
            max_values_numev = get_maxrange(case_numev)

        not_max_values_numev = []
        # detected_not_numev = []
        for level in tqdm(levels, desc = "Computing additional {} aggregations of the Number of Events Per Case (NEPC) for each of the {} most frequently occurring levels".format(frequency, max_k)):
            not_case_numev = case_log[case_log[case_feature]!=level].pivot_table(values = 'num_events', index = time_col, aggfunc = numeric_agg, fill_value = 0)
            not_case_numev.columns = ['NOT_'+level+'_numev']
            period_df = period_df.merge(not_case_numev, left_index = True, right_index= True, how = 'left')
            if xtr_outlier_rem:
                not_max_numev = get_maxrange(not_case_numev)
                not_max_values_numev.append(not_max_numev[0])

        plt_caseft_events_case()


    elif type == 'type_outcome':
        max_values_out = []
        not_max_values_out = []
        for level in tqdm(levels, desc = "Computing additional {} outcome aggregations for each of the {} most frequently occurring levels".format(frequency, max_k)):
            level_log = case_log[case_log[case_feature]==level]
            level_prcTrue = get_outcome_percentage(filtered_log= level_log, outcome = outcome, time_col = time_col)
            level_prcTrue.columns = [level+'_prc_True']
            period_df = period_df.merge(level_prcTrue, left_index= True, right_index= True, how= 'left')

            not_level_log = case_log[case_log[case_feature]!=level]
            not_level_prcTrue = get_outcome_percentage(filtered_log= not_level_log, outcome = outcome, time_col = time_col)
            not_level_prcTrue.columns = ['NOT_'+level+'_prc_True']
            period_df = period_df.merge(not_level_prcTrue, left_index= True, right_index= True, how= 'left')

            if xtr_outlier_rem:    
                max_out = get_maxrange(level_prcTrue)
                max_values_out.append(max_out[0])  

                not_max_out = get_maxrange(not_level_prcTrue)
                not_max_values_out.append(not_max_out[0])
            
        plt_caseft_outcome()

    plt.show()


######################################################################
###           EVOLUTION NUMERIC CASE FEATURES OVER TIME            ###
######################################################################

def num_casefts_evol(log, numeric_case_list, outcome = None, time_unit='days', frequency='weekly', case_assignment = 'first_event', type = 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    numeric_case_list : list of str
        Column names of the numerical case features. 
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.

    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)
    
    def plt_casefts_uni():
        fig, ax = plt.subplots(num_ftrs+2, 1)
        fig.set_size_inches([20, 6.25*(num_ftrs+2)])
        st = plt.suptitle("{} evolution of the {} of the numeric case features:".format(frequency, numeric_agg), fontsize=20)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e')

        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(num_ftrs):
            case_feature = numeric_case_list[i]
            #4
            per_series = period_df[case_feature]
            per_norm = (per_series - per_series.mean()) / per_series.std()
            plt_period(x, y = per_norm, axes = ax[1], y_label = "{} {} case features (Normalized)".format(frequency, numeric_agg), number = i+1, max_k=num_ftrs, 
                    title = "Numerical case features: {} {} (Normalized)".format(frequency, numeric_agg), 
                    label = "{}. '{}'".format(i+1, case_feature))
            #3
            plt_period(x, y = period_df[case_feature], axes=ax[i+2], y_label = "{} '{}'".format(numeric_agg, case_feature), 
                    title= "{}. '{}': {} evolution of the {}".format(i+1, case_feature, frequency, numeric_agg))

            if xtr_outlier_rem:
                ax[i+2].set_ylim(top = max_values_uni[i]*1.05)



        ax[num_ftrs+1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_casefts_tt():
        fig, ax = plt.subplots(num_ftrs+1, 2)
        fig.set_size_inches([20, 6.25*(num_ftrs+1)])
        st = plt.suptitle("{} evolution of the {} of the numeric case features, and of their relation with the {} Troughput Time (TT):".format(frequency,
             numeric_agg, numeric_agg), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )
        
        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(num_ftrs):
            case_feature = numeric_case_list[i]
            per_series = period_df[case_feature]
            per_norm = (per_series - per_series.mean()) / per_series.std()
            # Retrieving automatically determined ratio unit 
            ratio_unit = time_ratios[i]
            #4
            plt_period(x, y = per_norm, axes = ax[0,1], y_label = "{} {} case features (Normalized)".format(frequency, numeric_agg), number = i+1, max_k=num_ftrs, 
                    title = "Numerical case features: {} {} (Normalized)".format(frequency, numeric_agg), 
                    label = "{}. '{}'".format(i+1, case_feature))
            #3
            plt_period(x, y = period_df[case_feature], axes=ax[i+1,0], y_label = "{} '{}'".format(numeric_agg, case_feature), 
                    title= "{}. '{}': {} evolution of the {}".format(i+1, case_feature, frequency, numeric_agg))

            # Right:    Periodic ratio of (numeric_agg TT)/(numeric_agg case_feature)
            #3
            plt_period(x, y = period_df[case_feature+'_tt_ratio'], axes=ax[i+1,1], y_label = "({} TT) / (unit {})".format(ratio_unit, case_feature), 
                    title= "{} {} ratio of TT (in {}) per unit of {}".format(frequency, numeric_agg, ratio_unit, case_feature))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_uni[i] * 1.05)
                ax[i+1, 1].set_ylim(top = max_values_tt[i] * 1.05)

        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_casefts_events_case():
        fig, ax = plt.subplots(num_ftrs+1, 2)
        fig.set_size_inches([20, 6.25*(num_ftrs+1)])
        st = plt.suptitle("{} evolution of the {} of the numeric case features, and of their relation with the {} Number of Events Per Case (NEPC):".format(frequency,
             numeric_agg, numeric_agg), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of NEPC
        plt_period(x, period_df['num_events'], ax_0_r, y_label= "{} Number of Events Per Case (NEPC)".format(numeric_agg), label= "{} NEPC".format(numeric_agg),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases * 1.05)
            ax_0_r.set_ylim(top = max_global_numev * 1.05)

        #   All other plots: 
        for i in range(num_ftrs):
            case_feature = numeric_case_list[i]
            ratio_z = ratio_z_casefts[i]
            per_series = period_df[case_feature]
            per_norm = (per_series - per_series.mean()) / per_series.std()
            #4
            plt_period(x, y = per_norm, axes = ax[0,1], y_label = "{} {} case features (Normalized)".format(frequency, numeric_agg), number = i+1, max_k=num_ftrs, 
                    title = "Numerical case features: {} {} (Normalized)".format(frequency, numeric_agg), 
                    label = "{}. '{}'".format(i+1, case_feature))
            #3
            plt_period(x, y = period_df[case_feature], axes=ax[i+1,0], y_label = "{} '{}'".format(numeric_agg, case_feature), 
                    title= "{}. '{}': {} evolution of the {}".format(i+1, case_feature, frequency, numeric_agg))

            # Right:    Periodic ratio of (numeric_agg TT)/(numeric_agg case_feature)
            plt_period(x, y = period_df[case_feature+'_numev_ratio'], axes=ax[i+1,1], y_label = "(NEPC) / ({} {})".format(ratio_z, case_feature), 
                    title= "{} {} ratio of NEPC per {} of {}".format(frequency, numeric_agg, ratio_z, case_feature))

            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'numeric feature evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_uni[i]*1.05)
                # Accounting for possible outliers in right 'NEPC evolution plot'
                ax[i+1, 1].set_ylim(top = max_values_numev[i] * 1.05)

        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_casefts_outcome():
        fig, ax = plt.subplots(num_ftrs+1, 2)
        fig.set_size_inches([20, 6.25*(num_ftrs+1)])
        st = plt.suptitle("{} evolution of the {} of the numeric case features, and of their relation with outcome '{}':".format(frequency,
             numeric_agg, outcome), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic fraction of cases with outcome = True ( = 1)
        plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                location = 'right', color= '#ff7f0e' )
        
        if xtr_outlier_rem:
            # Accounting for possible outliers in evolution # cases initialized 
            ax[0,0].set_ylim(top = max_global_cases * 1.05)
            # Accounting for possible outliers in evolution fraction cases with outcome == True ( == 1)
            ax_0_r.set_ylim(top = max_global_out * 1.05)

        #   All other plots: 
        for i in range(num_ftrs):
            case_feature = numeric_case_list[i]
            per_series = period_df[case_feature]
            per_norm = (per_series - per_series.mean()) / per_series.std()
            #4
            plt_period(x, y = per_norm, axes = ax[0,1], y_label = "{} {} case features (Normalized)".format(frequency, numeric_agg), number = i+1, max_k=num_ftrs, 
                    title = "Numerical case features: {} {} (Normalized)".format(frequency, numeric_agg), 
                    label = "{}. '{}'".format(i+1, case_feature))
            #3
            plt_period(x, y = period_df[case_feature], axes=ax[i+1,0], y_label = "{} '{}'".format(numeric_agg, case_feature), 
                    title= "{}. '{}': {} evolution of the {}".format(i+1, case_feature, frequency, numeric_agg))

            plt_period(x, y= period_df[case_feature+'agg_True'], axes = ax[i+1, 1], y_label = "{} '{}'".format(numeric_agg, case_feature), number = 1, max_k=2, 
                    title = "'{}': {} evolution of the {} for '{}' = True vs = False".format(case_feature, frequency, numeric_agg, outcome),
                    label = "{} '{}' for cases with '{}' = True".format(numeric_agg, case_feature, outcome))
            plt_period(x, y= period_df[case_feature+'agg_False'], axes = ax[i+1, 1], y_label = "{} '{}'".format(numeric_agg, case_feature), number = 2, max_k=2, 
                    title = "'{}': {} evolution of the {} for '{}' = True vs = False".format(case_feature, frequency, numeric_agg, outcome),
                    label = "{} '{}' for cases with '{}' = False".format(numeric_agg, case_feature, outcome))
        
            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'numeric feature evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_uni[i] * 1.05)
                # Accounting for possible outliers in right 'Evolution feature for cases with outcome == 1 evolution plot'
                max_y = max(max_values_out[i], not_max_values_out[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)

        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def determ_ratio_NEPC_unit(ratio_ser_loc):
        z = 0
        while ratio_ser_loc.abs().mean()<1:
            ratio_ser_loc = ratio_ser_loc * 10
            z += 1
        return ratio_ser_loc, z

    case_log= log.drop_duplicates(subset='case:concept:name').copy()
    num_ftrs = len(numeric_case_list) #number of numeric case features

    # Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']

    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    # Adding periodic numeric_agg num_events
    if type == 'type_events_case':
        period_numev = case_log.pivot_table(values = 'num_events', index= time_col, aggfunc = numeric_agg, fill_value = 0)
        period_df = period_df.merge(period_numev, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_numev = get_maxrange(period_df[['num_events']])
            max_global_numev = max_global_numev[0]
    
    # Adding periodic fraction of cases with outcome = True (= 1)
    elif type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]


    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]

    x=period_df.index
    ratio_z_casefts = []

    # Periodic aggregations of all given numeric case features:
    period_caseft = case_log.pivot_table(values = numeric_case_list, index = time_col, aggfunc = numeric_agg, fill_value = 0)
    # Re-arranging columns:
    period_caseft = period_caseft[numeric_case_list]
    period_df = period_df.merge(period_caseft, left_index = True, right_index = True, how = 'left')
    if xtr_outlier_rem:
        max_values_uni = get_maxrange(period_caseft)

    if type == 'univariate':
        plt_casefts_uni()

    elif type == 'type_tt':
        period_ttr, time_ratios = get_tt_ratios(log = log, num_fts_list = numeric_case_list, time_col = time_col, numeric_agg = numeric_agg)
        period_df = period_df.merge(period_ttr, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem:
            max_values_tt = get_maxrange(period_ttr)
        plt_casefts_tt()

    elif type == 'type_events_case': 
        for case_feature in tqdm(numeric_case_list, 
                desc= "Computing the additional {} NEPC aggregations for each of the {} given numerical case features".format(frequency, num_ftrs)):  
            # Add periodic ratio [NEPC] / [10^(z) case_feature] with automatic determination of z
            case_log_pos = case_log[case_log[case_feature]!=0].dropna(subset=case_feature).copy()
            ratio_ser = case_log_pos['num_events'] / case_log_pos[case_feature]
            # Automatic determination of z
            ratio_z = 'unit'
            if ratio_ser.abs().mean() < 1:
                ratio_ser, z = determ_ratio_NEPC_unit(ratio_ser)
                ratio_z = "10^({}) units".format(z)
            ratio_z_casefts.append(ratio_z)
            case_log_pos[case_feature+'_numev_ratio'] = ratio_ser
            period_numev_ratio = case_log_pos.pivot_table(values = case_feature+'_numev_ratio', index = time_col, aggfunc= numeric_agg, fill_value = 0)
            period_df = period_df.merge(period_numev_ratio, left_index= True, right_index= True, how= 'left')
        if xtr_outlier_rem:
            numev_cols = [caseft + '_numev_ratio' for caseft in numeric_case_list]
            max_values_numev = get_maxrange(period_df[numev_cols])
        plt_casefts_events_case()
    
    elif type == 'type_outcome':
        for case_feature in tqdm(numeric_case_list, desc= "Computing the additional {} outcome aggregations for each of the {} given numerical case features".format(frequency, num_ftrs)):
            # Filtering out only the positive cases (outcome == 1)
            case_log_True = case_log[case_log[outcome]==1]
            period_True = case_log_True.dropna(subset=case_feature).pivot_table(values = case_feature, index = time_col, aggfunc = numeric_agg, fill_value=0)
            period_True.columns = [case_feature+'agg_True']
            period_df = period_df.merge(period_True, left_index= True, right_index= True, how= 'left')

            # Filtering out only the negative cases (outcome == 0)
            case_log_False = case_log[case_log[outcome]==0]
            period_False = case_log_False.dropna(subset=case_feature).pivot_table(values = case_feature, index = time_col, aggfunc = numeric_agg, fill_value=0)
            period_False.columns = [case_feature+'agg_False']
            period_df = period_df.merge(period_False, left_index= True, right_index= True, how= 'left')
        if xtr_outlier_rem:
            outTrue_cols = [caseft + 'agg_True' for caseft in numeric_case_list]
            outFalse_cols = [caseft + 'agg_False' for caseft in numeric_case_list]
            max_values_out = get_maxrange(period_df[outTrue_cols])
            not_max_values_out = get_maxrange(period_df[outFalse_cols])
        plt_casefts_outcome()

    plt.show()


#############################################################################################################################
#############################################################################################################################
###############                                                                                               ###############
###############                              EVOLUTION EVENT FEATURES OVER TIME                               ###############
###############                                                                                               ###############
#############################################################################################################################
#############################################################################################################################


######################################################################
###          EVOLUTION NUMERIC EVENT FEATURES OVER TIME            ###
######################################################################

def num_eventfts_evol(log, numeric_event_list, outcome = None, time_unit='days', frequency='weekly', case_assignment = 'first_event', type = 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True, numEventFt_transform = 'last'):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    numeric_event_list : list of str
        Column names of the numerical event features. 
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.
    numEventFt_transform : {'last', 'first', 'mean', 'median', 'sum', 'prod', 'min', 'max'}
        If any numeric event features contained in 'event_features', 'numEventFt_transform' 
        determines the way in which these numerical event features are transformed to the 
        case level. By default 'last'. 

    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """

    time_col = determine_time_col(frequency, case_assignment)
    
    tt_col = determine_tt_col(time_unit)
    


    def plt_eventfts_uni():
        fig, ax = plt.subplots(num_ftrs+2, 1)
        fig.set_size_inches([20, 6.25*(num_ftrs+2)])
        st = plt.suptitle("{} evolution of the {} of the numeric event features:".format(frequency, numeric_agg), fontsize=20)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(num_ftrs):
            event_feature_str = numeric_event_list_strings[i]
            event_feature = numeric_event_list[i]
            #4
            per_series = period_df[event_feature]
            per_norm = (per_series - per_series.mean()) / per_series.std()
            plt_period(x, y = per_norm, axes = ax[1], y_label = "{} {} event features (Normalized)".format(frequency, numeric_agg), number = i+1, max_k=num_ftrs, 
                    title = "Numerical event features: {} {} (Normalized)".format(frequency, numeric_agg), 
                    label = "{}. '{}'".format(i+1, event_feature_str))
            #3
            plt_period(x, y = period_df[event_feature], axes=ax[i+2], y_label = "{} '{}'".format(numeric_agg, event_feature_str), 
                    title= "{}. '{}': {} evolution of the {}".format(i+1, event_feature_str, frequency, numeric_agg))

            if xtr_outlier_rem:
                ax[i+2].set_ylim(top = max_values_uni[i]*1.05)

        ax[num_ftrs+1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_eventfts_tt():
        fig, ax = plt.subplots(num_ftrs+1, 2)
        fig.set_size_inches([20, 6.25*(num_ftrs+1)])
        st = plt.suptitle("{} evolution of the {} of the numeric event features, and of their relation with the {} Troughput Time (TT):".format(frequency,
             numeric_agg, numeric_agg), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(num_ftrs):
            event_feature_str = numeric_event_list_strings[i]
            event_feature = numeric_event_list[i]
            per_series = period_df[event_feature]
            per_norm = (per_series - per_series.mean()) / per_series.std()
            # Retrieving automatically determined ratio unit 
            ratio_unit = time_ratios[i]
            #4
            plt_period(x, y = per_norm, axes = ax[0,1], y_label = "{} {} event features (Normalized)".format(frequency, numeric_agg), number = i+1, max_k=num_ftrs, 
                    title = "Numerical event features: {} {} (Normalized)".format(frequency, numeric_agg), 
                    label = "{}. '{}'".format(i+1, event_feature_str))
            #3
            plt_period(x, y = period_df[event_feature], axes=ax[i+1,0], y_label = "{} '{}'".format(numeric_agg, event_feature_str), 
                    title= "{}. '{}': {} evolution of the {}".format(i+1, event_feature_str, frequency, numeric_agg))

            # Right:    Periodic ratio of (numeric_agg TT)/(numeric_agg case_feature)

            #3
            plt_period(x, y = period_df[event_feature+'_tt_ratio'], axes=ax[i+1,1], y_label = "({} TT) / (unit {})".format(ratio_unit, event_feature_str), 
                    title= "{} {} ratio of TT (in {}) per unit of {}".format(frequency, numeric_agg, ratio_unit, event_feature_str))
                
            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_uni[i] * 1.05)
                ax[i+1, 1].set_ylim(top = max_values_tt[i] * 1.05)

        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_eventfts_events_case():
        fig, ax = plt.subplots(num_ftrs+1, 2)
        fig.set_size_inches([20, 6.25*(num_ftrs+1)])
        st = plt.suptitle("{} evolution of the {} of the numeric event features, and of their relation with the {} Number of Events Per Case (NEPC):".format(frequency,
             numeric_agg, numeric_agg), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of NEPC
        plt_period(x, period_df['num_events'], ax_0_r, y_label= "{} Number of Events Per Case (NEPC)".format(numeric_agg), label= "{} NEPC".format(numeric_agg),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases * 1.05)
            ax_0_r.set_ylim(top = max_global_numev * 1.05)

        #   All other plots: 
        for i in range(num_ftrs):
            event_feature_str = numeric_event_list_strings[i]
            event_feature = numeric_event_list[i]
            ratio_z = ratio_z_casefts[i]
            per_series = period_df[event_feature]
            per_norm = (per_series - per_series.mean()) / per_series.std()
            #4
            plt_period(x, y = per_norm, axes = ax[0,1], y_label = "{} {} event features (Normalized)".format(frequency, numeric_agg), number = i+1, max_k=num_ftrs, 
                    title = "Numerical event features: {} {} (Normalized)".format(frequency, numeric_agg), 
                    label = "{}. '{}'".format(i+1, event_feature_str))
            #3
            plt_period(x, y = period_df[event_feature], axes=ax[i+1,0], y_label = "{} '{}'".format(numeric_agg, event_feature_str), 
                    title= "{}. '{}': {} evolution of the {}".format(i+1, event_feature_str, frequency, numeric_agg))

            # Right:    Periodic ratio of (numeric_agg TT)/(numeric_agg case_feature)
            plt_period(x, y = period_df[event_feature+'_numev_ratio'], axes=ax[i+1,1], y_label = "(NEPC) / ({} {})".format(ratio_z, event_feature_str), 
                    title= "{} {} ratio of NEPC per {} of {}".format(frequency, numeric_agg, ratio_z, event_feature_str))
            
            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'numeric feature evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_uni[i]*1.05)
                # Accounting for possible outliers in right 'NEPC evolution plot'
                ax[i+1, 1].set_ylim(top = max_values_numev[i] * 1.05)

        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_eventfts_outcome():
        fig, ax = plt.subplots(num_ftrs+1, 2)
        fig.set_size_inches([20, 6.25*(num_ftrs+1)])
        st = plt.suptitle("{} evolution of the {} of the numeric event features, and of their relation with outcome '{}':".format(frequency,
             numeric_agg, outcome), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic fraction of cases with outcome = True ( = 1)
        plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            # Accounting for possible outliers in evolution # cases initialized 
            ax[0,0].set_ylim(top = max_global_cases * 1.05)
            # Accounting for possible outliers in evolution fraction cases with outcome == True ( == 1)
            ax_0_r.set_ylim(top = max_global_out * 1.05)

        #   All other plots: 
        for i in range(num_ftrs):
            event_feature_str = numeric_event_list_strings[i]
            event_feature = numeric_event_list[i]
            per_series = period_df[event_feature]
            per_norm = (per_series - per_series.mean()) / per_series.std()
            #4
            plt_period(x, y = per_norm, axes = ax[0,1], y_label = "{} {} event features (Normalized)".format(frequency, numeric_agg), number = i+1, max_k=num_ftrs, 
                    title = "Numerical event features: {} {} (Normalized)".format(frequency, numeric_agg), 
                    label = "{}. '{}'".format(i+1, event_feature_str))
            #3
            plt_period(x, y = period_df[event_feature], axes=ax[i+1,0], y_label = "{} '{}'".format(numeric_agg, event_feature_str), 
                    title= "{}. '{}': {} evolution of the {}".format(i+1, event_feature_str, frequency, numeric_agg))

            plt_period(x, y= period_df[event_feature+'agg_True'], axes = ax[i+1, 1], y_label = "{} '{}'".format(numeric_agg, event_feature_str), number = 1, max_k=2, 
                    title = "'{}': {} evolution of the {} for '{}' = True vs = False".format(event_feature_str, frequency, numeric_agg, outcome),
                    label = "{} '{}' for cases with '{}' = True".format(numeric_agg, event_feature_str, outcome))
            plt_period(x, y= period_df[event_feature+'agg_False'], axes = ax[i+1, 1], y_label = "{} '{}'".format(numeric_agg, event_feature_str), number = 2, max_k=2, 
                    title = "'{}': {} evolution of the {} for '{}' = True vs = False".format(event_feature_str, frequency, numeric_agg, outcome),
                    label = "{} '{}' for cases with '{}' = False".format(numeric_agg, event_feature_str, outcome))

            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'numeric feature evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_uni[i] * 1.05)
                # Accounting for possible outliers in right 'Evolution feature for cases with outcome == 1 evolution plot'
                max_y = max(max_values_out[i], not_max_values_out[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)

        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[num_ftrs, 0].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def determ_ratio_NEPC_unit(ratio_ser_loc):
        z = 0
        while ratio_ser_loc.abs().mean()<1:
            ratio_ser_loc = ratio_ser_loc * 10
            z += 1
        return ratio_ser_loc, z
    local_log = log.copy()
    local_log = _event_fts_to_tracelvl(local_log, event_features = numeric_event_list, numEventFt_transform = numEventFt_transform)
    case_log= local_log.drop_duplicates(subset='case:concept:name').copy()
    # Numeric event features are brought back to the trace level in a preprocessing step:
    numeric_event_list_strings = [event_ftr for event_ftr in numeric_event_list]
    numeric_event_list= [event_ftr+'_trace' for event_ftr in numeric_event_list]
    num_ftrs = len(numeric_event_list) #number of numeric event features
    # Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']

    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    # Adding periodic numeric_agg num_events
    if type == 'type_events_case':
        period_numev = case_log.pivot_table(values = 'num_events', index= time_col, aggfunc = numeric_agg, fill_value = 0)
        period_df = period_df.merge(period_numev, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_numev = get_maxrange(period_df[['num_events']])
            max_global_numev = max_global_numev[0]
    
    # Adding periodic fraction of cases with outcome = True (= 1)
    elif type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]


    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]

    x=period_df.index
    ratio_z_casefts = []

    # Periodic aggregations of all given numeric event features:
    period_caseft = case_log.pivot_table(values = numeric_event_list, index = time_col, aggfunc = numeric_agg, fill_value = 0)
    # Re-arranging columns:
    period_caseft = period_caseft[numeric_event_list]
    period_df = period_df.merge(period_caseft, left_index = True, right_index = True, how = 'left')
    if xtr_outlier_rem:
        max_values_uni = get_maxrange(period_caseft)

    if type == 'univariate':
        plt_eventfts_uni()

    elif type == 'type_tt':
        period_ttr, time_ratios = get_tt_ratios(log = local_log, num_fts_list = numeric_event_list, time_col = time_col, numeric_agg = numeric_agg)
        period_df = period_df.merge(period_ttr, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem:
            max_values_tt = get_maxrange(period_ttr)
        plt_eventfts_tt()
    
    elif type == 'type_events_case': 
        for event_feature in tqdm(numeric_event_list, 
                desc= "Computing the additional {} NEPC aggregations for each of the {} given numerical event features".format(frequency, num_ftrs)):  
            # Add periodic ratio [NEPC] / [10^(z) event_feature] with automatic determination of z
            case_log_pos = case_log[case_log[event_feature]!=0].dropna(subset=event_feature).copy()
            ratio_ser = case_log_pos['num_events'] / case_log_pos[event_feature]
            # Automatic determination of z
            ratio_z = 'unit'
            if ratio_ser.abs().mean() < 1:
                ratio_ser, z = determ_ratio_NEPC_unit(ratio_ser)
                ratio_z = "10^({}) units".format(z)
            ratio_z_casefts.append(ratio_z)
            case_log_pos[event_feature+'_numev_ratio'] = ratio_ser
            period_numev_ratio = case_log_pos.pivot_table(values = event_feature+'_numev_ratio', index = time_col, aggfunc= numeric_agg, fill_value = 0)
            period_df = period_df.merge(period_numev_ratio, left_index= True, right_index= True, how= 'left')
        if xtr_outlier_rem:
            numev_cols = [eventft + '_numev_ratio' for eventft in numeric_event_list]
            max_values_numev = get_maxrange(period_df[numev_cols])
        plt_eventfts_events_case()

    elif type == 'type_outcome':
        for event_feature in tqdm(numeric_event_list, desc= "Computing the additional {} outcome aggregations for each of the {} given numerical event features".format(frequency, num_ftrs)):
            # Filtering out only the positive cases (outcome == 1)
            case_log_True = case_log[case_log[outcome]==1]
            period_True = case_log_True.dropna(subset = event_feature).pivot_table(values = event_feature, index = time_col, aggfunc = numeric_agg, fill_value=0)
            period_True.columns = [event_feature+'agg_True']
            period_df = period_df.merge(period_True, left_index= True, right_index= True, how= 'left')

            # Filtering out only the negative cases (outcome == 0)
            case_log_False = case_log[case_log[outcome]==0]
            period_False = case_log_False.dropna(subset= event_feature).pivot_table(values = event_feature, index = time_col, aggfunc = numeric_agg, fill_value=0)
            period_False.columns = [event_feature+'agg_False']
            period_df = period_df.merge(period_False, left_index= True, right_index= True, how= 'left')
        if xtr_outlier_rem:
            outTrue_cols = [eventft + 'agg_True' for eventft in numeric_event_list]
            outFalse_cols = [eventft + 'agg_False' for eventft in numeric_event_list]
            max_values_out = get_maxrange(period_df[outTrue_cols])
            not_max_values_out = get_maxrange(period_df[outFalse_cols])
        plt_eventfts_outcome()

    plt.show()

######################################################################
###        EVOLUTION CATEGORICAL EVENT FEATURES OVER TIME          ###
######################################################################

def topK_categorical_eventftr_evol(log, event_feature, outcome = None, time_unit = 'days', frequency = 'weekly', case_assignment = 'first_event', plt_type = 'univariate', numeric_agg = 'mean', max_k = 10, xtr_outlier_rem = True):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    event_feature : str
        Column name of the categorical event feature.
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    plt_type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'plt_type' options, see Notes.
    numeric_agg : str, optional
        _description_, by default 'mean'
    max_k : int, optional
        Only the 'max_k' most frequently occurring levels of the feature are considered, by default 10. 
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.

    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    # Inner functions belonging to a specific plot type:
    def plt_eventft_uni():
        fig, ax = plt.subplots(max_k+2, 1)
        fig.set_size_inches([20, 6.25*(max_k+2)])
        st = plt.suptitle("{}: {} evolution fractions for each of the {} most frequent levels".format(event_feature, frequency, max_k), fontsize=15)
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k):
            level = levels[i]
            level_string = level_strings[i]
            #4
            plt_period(x, y = period_df[level_string+'_prc'], axes = ax[1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} fraction of cases belonging to the {} most common levels of '{}'".format(frequency, max_k, event_feature), label = '{}. {}'.format(i+1, level))
            #3
            plt_period(x, y = period_df[level_string+'_prc'], axes=ax[i+2], y_label = level+": fraction cases", 
                    title= "{}. '{}' = '{}' (at least once): {} evolution fraction cases".format(i+1, event_feature, level, frequency))

            if xtr_outlier_rem:
                ax[i+2].set_ylim(top = max_values_prc[i]*1.05)
        if xtr_outlier_rem:
            ax[1].set_ylim(top = np.nanmax(max_values_prc)*1.05)
    
        ax[max_k+1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)


    def plt_eventft_tt():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{}: {} evolution fractions and {} Throughput Time (TT) for each of the {} most frequent levels".format(event_feature, frequency,
                        numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of TT
        plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_tt*1.05)

        #   All other plots: 
        for i in range(max_k):
            level = levels[i]
            level_string = level_strings[i]
            #4
            plt_period(x, y=period_df[level_string+'_prc'], axes = ax[0, 1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} fraction of cases belonging to the {} most common levels of '{}'".format(frequency, max_k, event_feature), label = level)
            #3
            plt_period(x, y= period_df[level_string+'_prc'], axes=ax[i+1, 0], y_label = level+": fraction cases", 
                    title= "{}. '{}' = '{}' (at least once): {} evolution fraction cases".format(i+1, event_feature, level, frequency))
            #4
            plt_period(x, y= period_df[level_string+"_tt"], axes = ax[i+1, 1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 1, max_k=2, 
                    title = "{}. '{}' = '{}' (at least once): {} evolution {} TT".format(i+1, event_feature, level, frequency, numeric_agg),
                    label = "{} TT ({}) for cases with '{}' = '{}' (at least once)".format(numeric_agg, time_unit, event_feature, level))
            plt_period(x, y= period_df["NOT_"+level_string+"_tt"], axes = ax[i+1, 1], y_label = "{} TT ({})".format(numeric_agg, time_unit), number = 2, max_k=2, 
                    title = "{}. '{}' = '{}' (at least once): {} evolution {} TT".format(i+1, event_feature, level, frequency, numeric_agg),
                    label = "{} TT ({}) for cases with '{}' NOT = '{}'".format(numeric_agg, time_unit, event_feature, level))

            if xtr_outlier_rem:
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                max_y = max(max_values_tt[i], not_max_values_tt[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    def plt_eventft_events_case(): 
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{}: {} evolution fractions and {} Number of Events Per Case (NEPC) for each of the {} most frequent levels".format(event_feature, frequency,
                        numeric_agg, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic numeric_agg of NEPC
        plt_period(x, period_df['num_events'], ax_0_r, y_label= "{} Number of Events Per Case (NEPC)".format(numeric_agg), label= "{} NEPC".format(numeric_agg),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            ax_0_r.set_ylim(top = max_global_numev*1.05)

        #   All other plots: 
        for i in range(max_k):
            level = levels[i]
            level_string = level_strings[i]
            #4
            plt_period(x, y=period_df[level_string+'_prc'], axes = ax[0, 1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} fraction of cases belonging to the {} most common levels of '{}'".format(frequency, max_k, event_feature), label = level)
            #3
            plt_period(x, y=period_df[level_string+'_prc'], axes=ax[i+1, 0], y_label = level+": fraction cases", 
                    title= "{}. '{}' = '{}' (at least once): {} evolution fraction cases".format(i+1, event_feature, level, frequency))

            #4
            plt_period(x, y= period_df[level_string+"_numev"], axes = ax[i+1, 1], y_label = "{} Number Events Per Case (NEPC)".format(numeric_agg), number = 1, max_k=2, 
                    title = "{}. '{}' = '{}' (at least once): {} evolution {} NEPC".format(i+1, event_feature, level, frequency, numeric_agg),
                    label = "{} NEPC for cases with '{}' = '{}' (at least once)".format(numeric_agg, event_feature, level))
            plt_period(x, y= period_df["NOT_"+level_string+"_numev"], axes = ax[i+1, 1], y_label = "{} Number Events Per Case (NEPC)".format(numeric_agg), number = 2, max_k=2, 
                    title = "{}. '{}' = '{}' (at least once): {} evolution {} NEPC".format(i+1, event_feature, level, frequency, numeric_agg),
                    label = "{} NEPC for cases with '{}' NOT = '{}'".format(numeric_agg, event_feature, level))

            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'fraction evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                # Accounting for possible outliers in right 'NEPC evolution plot'
                max_y = max(max_values_numev[i], not_max_values_numev[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)
        if xtr_outlier_rem:
            # Accounting for possible outliers in global 'fraction evolution plot'
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)
    
    def plt_eventft_outcome():
        fig, ax = plt.subplots(max_k+1, 2)
        fig.set_size_inches([20, 6.25*(max_k+1)])
        st = plt.suptitle("{}: {} evolution fractions and fraction of cases with '{}' = True for each of the {} most frequent levels".format(event_feature, frequency,
                        outcome, max_k), fontsize=15)
        
        #   First plot:
        #       - periodic # cases initialized
        plt_period(x, period_df['total'], ax[0,0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), location = 'left', color = '#1f77b4')
        ax_0_r= ax[0,0].twinx()
        #       - periodic fraction of cases with outcome = True ( = 1)
        plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                location = 'right', color= '#ff7f0e' )

        if xtr_outlier_rem:
            # Accounting for possible outliers in evolution # cases initialized 
            ax[0,0].set_ylim(top = max_global_cases*1.05)
            # Accounting for possible outliers in evolution fraction cases with outcome == True ( == 1)
            ax_0_r.set_ylim(top = max_global_out*1.05)

        #   All other plots: 
        for i in range(max_k):
            level = levels[i]
            level_string = level_strings[i]
            #4
            plt_period(x, y=period_df[level_string+'_prc'], axes = ax[0, 1], y_label = "Fraction of initialized cases", number = i+1, max_k=max_k, 
                    title = "{} fraction of cases belonging to the {} most common levels of '{}'".format(frequency, max_k, event_feature), label = level)
            #3
            plt_period(x, y=period_df[level_string+'_prc'], axes=ax[i+1, 0], y_label = level+": fraction cases", 
                    title= "{}. '{}' = '{}' (at least once): {} evolution fraction cases".format(i+1, event_feature, level, frequency))

            #4
            plt_period(x, y= period_df[level_string+'_prc_True'], axes = ax[i+1, 1], y_label = "Fraction cases '{}' = True".format(outcome), number = 1, max_k=2, 
                    title = "{}. '{}' = '{}' (at least once): {} evolution fraction '{}' = True".format(i+1, event_feature, level, frequency, outcome),
                    label = "Fraction '{}' = True for cases with '{}' = '{}' (at least once)".format(outcome, event_feature, level))
            plt_period(x, y= period_df['NOT_'+level_string+'_prc_True'], axes = ax[i+1, 1], y_label = "Fraction cases '{}' = True".format(outcome), number = 2, max_k=2, 
                    title = "{}. '{}' = '{}' (at least once): {} evolution fraction '{}' = True".format(i+1, event_feature, level, frequency, outcome),
                    label = "Fraction '{}' = True for cases with '{}' NOT = '{}'".format(outcome, event_feature, level))

            if xtr_outlier_rem:
                # Accounting for possible outliers in left 'fraction evolution plot'
                ax[i+1, 0].set_ylim(top = max_values_prc[i]*1.05)
                # Accounting for possible outliers in right 'Fraction cases with outcome == 1 evolution plot'
                max_y = max(max_values_out[i], not_max_values_out[i])
                ax[i+1, 1].set_ylim(top = max_y * 1.05)

        if xtr_outlier_rem:
            # Accounting for possible outliers in global 'fraction evolution plot'
            ax[0, 1].set_ylim(top = np.nanmax(max_values_prc)*1.05)

        ax[max_k, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[max_k, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        st.set_y(1)

    local_log = log.copy()

    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)

    levels = list(local_log.dropna(subset=event_feature)[event_feature].value_counts().index) # sorted
    num_levels = len(levels)
    if max_k < num_levels: 
        levels = levels[:max_k]
    elif max_k > num_levels:
        max_k = num_levels
        levels = levels[:max_k]
    #Getting the corresponding binary column names constructed during preprocessing: 
    level_strings = []
    for level in levels:
        # filt_df_cases = df[df[col] == val][case_id_key].unique()
        if type(level)!=bool:
            level_string = event_feature + "_" + level.encode('ascii', errors='ignore').decode('ascii').replace(" ", "")
            level_strings.append(level_string)
        elif type(level)==bool:
            if level:
                level_string = event_feature + "_" + "True"
            else:
                level_string = event_feature + "_" + "False"
            level_strings.append(level_string)
    # Adding the needed binary columns (1 for each level of the categorical event feature):
    local_log[event_feature] = np.where(local_log[event_feature].isin(levels), local_log[event_feature], 'OTHERS')
    local_log = _event_fts_to_tracelvl(local_log, event_features = [event_feature])

    levels = [str(level) for level in levels]

    case_log = local_log.drop_duplicates(subset = 'case:concept:name').copy()

    # Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']
    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    # Adding periodic numeric_agg num_events
    if plt_type == 'type_events_case':
        period_numev = case_log.pivot_table(values = 'num_events', index= time_col, aggfunc = numeric_agg, fill_value = 0)
        period_df = period_df.merge(period_numev, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_numev = get_maxrange(period_df[['num_events']])
            max_global_numev = max_global_numev[0]
    
    # Adding periodic fraction of cases with outcome = True (= 1)
    elif plt_type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]


    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]

    x = period_df.index

    # # Computing the needed values for the max_k most frequent levels:

    for i in tqdm(range(max_k), desc= "Computing the {} aggregations for each of the {} most frequently occurring levels of categorical event feature '{}'".format(frequency, max_k, event_feature)): 
        # level = levels[i]
        level_string = level_strings[i]
        # Computing periodic counts of each level
        case_sliced = case_log[case_log[level_string]==1]
        period_level = case_sliced.pivot_table(values = 'case:concept:name', index= time_col, aggfunc = 'count', fill_value=0)
        period_level.columns = [level_string]
        period_df = period_df.merge(period_level, left_index = True, right_index = True, how= 'left')
        # Computing periodic fractions of cases in which that level occurs 
        period_df[level_string+'_prc'] = period_df[level_string] / period_df['total']

        if plt_type == 'type_tt':
            # Computing periodic tt aggregations for cases in which that event feature level does occur:
            period_tt = case_sliced.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
            period_tt.columns = [level_string+'_tt']
            period_df = period_df.merge(period_tt, left_index = True, right_index = True, how = 'left')
            # Computing periodic tt aggregations for cases in which that event feature level does NOT occur:
            not_case_sliced = case_log[case_log[level_string]==0]
            # Exception handling: for in case that a certain categorical event feature level occurs in every case.
            num_cases = len(not_case_sliced)
            if num_cases == 0:
                period_df['NOT_'+level_string+'_tt'] = [np.nan for _ in range(len(period_df))]
            else:
                not_period_tt = not_case_sliced.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
                not_period_tt.columns = ['NOT_'+level_string+'_tt']
                period_df = period_df.merge(not_period_tt, left_index = True, right_index = True, how = 'left')
        
        elif plt_type == 'type_events_case':
            # Computing periodic NEPC aggregations for cases in which that event feature level does occur:
            period_numev = case_sliced.pivot_table(values = 'num_events', index = time_col, aggfunc = numeric_agg, fill_value = 0)
            period_numev.columns = [level_string + '_numev']
            period_df = period_df.merge(period_numev, left_index = True, right_index = True, how = 'left')
            # Computing periodic NEPC aggregations for cases in which that event feature level does NOT occur:
            not_case_sliced = case_log[case_log[level_string]==0]
            # Exception handling: for in case that a certain categorical event feature level occurs in every case.
            num_cases = len(not_case_sliced)
            if num_cases == 0:
                period_df['NOT_' + level_string + '_numev'] = [np.nan for _ in range(len(period_df))]
            else:
                not_period_numev = not_case_sliced.pivot_table(values= 'num_events', index= time_col, aggfunc = numeric_agg, fill_value=0) 
                not_period_numev.columns = ['NOT_' + level_string + '_numev']
                period_df = period_df.merge(not_period_numev, left_index = True, right_index = True, how = 'left')
        
        elif plt_type == 'type_outcome':
            # Computing periodic outcome aggregations for cases in which that event feature level does occur:
            period_prcTrue = get_outcome_percentage(filtered_log= case_sliced, outcome = outcome, time_col = time_col)
            period_prcTrue.columns = [level_string + '_prc_True']
            period_df = period_df.merge(period_prcTrue, left_index = True, right_index = True, how = 'left')
            # Computing periodic outcome aggregations for cases in which that event feature level does NOT occur:
            not_case_sliced = case_log[case_log[level_string]==0]
            # Exception handling: for in case that a certain categorical event feature level occurs in every case.
            num_cases = len(not_case_sliced)
            if num_cases == 0:
                period_df['NOT_' + level_string + '_prc_True'] = [np.nan for _ in range(len(period_df))]
            else:
                not_period_prcTrue = get_outcome_percentage(filtered_log= not_case_sliced, outcome = outcome, time_col = time_col)
                not_period_prcTrue.columns = ['NOT_' + level_string + '_prc_True']
                period_df = period_df.merge(not_period_prcTrue, left_index = True, right_index = True, how = 'left')
    if xtr_outlier_rem: 
        cols_prc = [level_string+'_prc' for level_string in level_strings]
        max_values_prc = get_maxrange(period_df[cols_prc])
        if plt_type == 'type_tt':
            cols_tt = [level_string+'_tt' for level_string in level_strings]
            not_cols_tt = ['NOT_'+level_string+'_tt' for level_string in level_strings]
            max_values_tt = get_maxrange(period_df[cols_tt])
            not_max_values_tt = get_maxrange(period_df[not_cols_tt])
        elif plt_type == 'type_events_case':
            cols_numev = [level_string+'_numev' for level_string in level_strings]
            not_cols_numev = ['NOT_'+level_string+'_numev' for level_string in level_strings]
            max_values_numev = get_maxrange(period_df[cols_numev])
            not_max_values_numev = get_maxrange(period_df[not_cols_numev])
        elif plt_type == 'type_outcome':
            cols_out = [level_string+'_prc_True' for level_string in level_strings]
            not_cols_out = ['NOT_'+ level_string+ '_prc_True' for level_string in level_strings]
            max_values_out = get_maxrange(period_df[cols_out])
            not_max_values_out = get_maxrange(period_df[not_cols_out])
    
    # Plotting: 
    if plt_type == 'univariate':
        plt_eventft_uni()
    elif plt_type == 'type_tt':
        plt_eventft_tt()
    elif plt_type == 'type_events_case': 
        plt_eventft_events_case()
    elif plt_type == 'type_outcome':
        plt_eventft_outcome()

    plt.show()


############################################################################################
###                            DISTINCT VARIANTS OVER TIME                               ###
############################################################################################



def distinct_variants_evol(log, outcome = None, time_unit='days', frequency='weekly', case_assignment = 'first_event', type= 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True, cases_initialized = True):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.
    cases_initialized : bool, optional
        _description_, by default True

    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    # Inner functions belonging to a specific plot type:
    def plt_distVars_uni():
        fig, ax = plt.subplots()
        fig.set_size_inches([20, 6.25])
        # Distinct variants evolution
        plt_period(x, period_df['num_distinct_vars'], ax, y_label = "Number Distinct Variants", label = "Number of Distinct Variants ({})".format(frequency), 
                    location = 'left', color = '#1f77b4', title = title_vars)
        ax_0_r= ax.twinx()
        # New distinct variants evolution 
        plt_period(x, period_df['num_NEW_distinct_vars'], ax_0_r, y_label= "Number Distinct New Variants", label= "Number of Distinct New Variants ({})".format(frequency),
                location = 'right', color= '#ff7f0e' )
        if xtr_outlier_rem:
            ax.set_ylim(top = max_global_varcounts[0]*1.05)
            ax_0_r.set_ylim(top = max_global_varcounts[1]*1.05)

        ax.set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        # st.set_y(1)

    def plt_distVars_tt():
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches([20, 6.25*2])
        if cases_initialized:
            title_ci = "Number of Initialized Cases (left) and {} Throughput Time (in {}) (right) over time".format(numeric_agg, time_unit)
            plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), 
                        location = 'left', color = '#1f77b4', title = title_ci)
            ax_0_r= ax[0].twinx()
            #       - periodic numeric_agg of TT
            plt_period(x, period_df[tt_col], ax_0_r, y_label= "{} Throughput Time".format(numeric_agg), label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                    location = 'right', color= '#ff7f0e')

            if xtr_outlier_rem:
                ax[0].set_ylim(top = max_global_cases*1.05)
                ax_0_r.set_ylim(top = max_global_tt*1.05)
        else: 
            #       - periodic numeric_agg of TT
            title_tt = "{} Throughput Time (in {}) over time".format(numeric_agg, time_unit)
            plt_period(x, period_df[tt_col], ax[0], y_label= "{} Throughput Time".format(numeric_agg), title = title_tt)
            if xtr_outlier_rem:
                ax[0].set_ylim(top = max_global_tt*1.05)

        # Distinct count plots:
        plt_period(x, period_df['num_distinct_vars'], ax[1], y_label = "Number Distinct Variants", label = "Number of Distinct Variants ({})".format(frequency), 
                    location = 'left', color = '#1f77b4', title = title_vars)
        ax_0_r= ax[1].twinx()
        # New distinct variants evolution 
        plt_period(x, period_df['num_NEW_distinct_vars'], ax_0_r, y_label= "Number Distinct New Variants", label= "Number of Distinct New Variants ({})".format(frequency),
                location = 'right', color= '#ff7f0e' )
        if xtr_outlier_rem:
            ax[1].set_ylim(top = max_global_varcounts[0]*1.05)
            ax_0_r.set_ylim(top = max_global_varcounts[1]*1.05)

        ax[1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
    
    def plt_distVars_events_case():
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches([20, 6.25*2])
        if cases_initialized:
            title_ci = "Number of Initialized Cases (left) and {} Number of Events Per Case (NEPC) (right) over time".format(numeric_agg)
            plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), 
                        location = 'left', color = '#1f77b4', title = title_ci)
            ax_0_r= ax[0].twinx()
            #       - periodic numeric_agg of NEPC
            plt_period(x, period_df['num_events'], ax_0_r, y_label= "{} Number of Events Per Case (NEPC)".format(numeric_agg), label= "{} NEPC".format(numeric_agg),
                    location = 'right', color= '#ff7f0e' )

            if xtr_outlier_rem:
                ax[0].set_ylim(top = max_global_cases*1.05)
                ax_0_r.set_ylim(top = max_global_numev*1.05)
        else: 
            #       - periodic numeric_agg of NEPC
            title_nepc = "{} Number of Events Per Case (NEPC) over time".format(numeric_agg)
            plt_period(x, period_df['num_events'], ax[0], y_label= "{} Number of Events Per Case (NEPC)".format(numeric_agg), title = title_nepc)
            if xtr_outlier_rem:
                ax[0].set_ylim(top = max_global_numev*1.05)

        # Distinct count plots:
        plt_period(x, period_df['num_distinct_vars'], ax[1], y_label = "Number Distinct Variants", label = "Number of Distinct Variants ({})".format(frequency), 
                    location = 'left', color = '#1f77b4', title = title_vars)
        ax_0_r= ax[1].twinx()
        # New distinct variants evolution 
        plt_period(x, period_df['num_NEW_distinct_vars'], ax_0_r, y_label= "Number Distinct New Variants", label= "Number of Distinct New Variants ({})".format(frequency),
                location = 'right', color= '#ff7f0e' )
        if xtr_outlier_rem:
            ax[1].set_ylim(top = max_global_varcounts[0]*1.05)
            ax_0_r.set_ylim(top = max_global_varcounts[1]*1.05)

        ax[1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
    
    def plt_distVars_outcome():
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches([20, 6.25*2])

        if cases_initialized:
            title_ci = "Number of Initialized Cases (left) and Fraction of cases with '{}' = True (right) over time".format(numeric_agg, outcome)
            plt_period(x, period_df['total'], ax[0], y_label = "# Cases", label = "# cases contained ({})".format(frequency), 
                        location = 'left', color = '#1f77b4', title = title_ci)
            ax_0_r= ax[0].twinx()
            #       - periodic fraction of cases with outcome = True ( = 1)
            plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= "Fraction outcome = True", label= "Fraction outcome '{}' = True".format(outcome),
                    location = 'right', color= '#ff7f0e' )

            if xtr_outlier_rem:
                ax[0].set_ylim(top = max_global_cases*1.05)
                ax_0_r.set_ylim(top = max_global_out*1.05)
        else: 
            title_out = "Fraction of cases with '{}' = True over time".format(outcome)
            plt_period(x, y= period_df['prc_True'], axes= ax[0], y_label= "Fraction outcome = True", title = title_out)
            if xtr_outlier_rem:
                ax[0].set_ylim(top = max_global_out*1.05)
        # Distinct count plots:
        plt_period(x, period_df['num_distinct_vars'], ax[1], y_label = "Number Distinct Variants", label = "Number of Distinct Variants ({})".format(frequency), 
                    location = 'left', color = '#1f77b4', title = title_vars)
        ax_0_r= ax[1].twinx()
        # New distinct variants evolution 
        plt_period(x, period_df['num_NEW_distinct_vars'], ax_0_r, y_label= "Number Distinct New Variants", label= "Number of Distinct New Variants ({})".format(frequency),
                location = 'right', color= '#ff7f0e' )
        if xtr_outlier_rem:
            ax[1].set_ylim(top = max_global_varcounts[0]*1.05)
            ax_0_r.set_ylim(top = max_global_varcounts[1]*1.05)

        ax[1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()

            
    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)

    local_log = log.copy()
    # Get dataframe containing the case id and variant (as a tuple of activity strings) for each case:
    case_variant = get_variant_case(log)
    local_log = local_log.merge(case_variant, on = 'case:concept:name', how = 'left')

    period_varcounts = get_uniq_varcounts(log = local_log, time_col = time_col)
    if xtr_outlier_rem:
        max_global_varcounts = get_maxrange(period_varcounts[['num_distinct_vars', 'num_NEW_distinct_vars']])
        

    case_log = log.drop_duplicates(subset='case:concept:name').copy()
    #Periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']
    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]

    # Adding periodic numeric_agg num_events
    if type == 'type_events_case':
        period_numev = case_log.pivot_table(values = 'num_events', index= time_col, aggfunc = numeric_agg, fill_value = 0)
        period_df = period_df.merge(period_numev, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_numev = get_maxrange(period_df[['num_events']])
            max_global_numev = max_global_numev[0]

    
    # Adding periodic fraction of cases with outcome = True (= 1)
    if type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]

    # Adding periodic numeric_agg tt 
    else: 
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]

    # Adding varcounts 
    period_df = period_df.merge(period_varcounts, left_index = True, right_index = True, how = 'left')

    
    x = period_df.index
    title_vars = "Number of Distinct Variants and New Distinct Variants over time"

    if type == 'univariate':
        plt_distVars_uni()

    elif type == 'type_tt':
        plt_distVars_tt()
    
    elif type == 'type_events_case':
        plt_distVars_events_case()

    elif type == 'type_outcome':
        plt_distVars_outcome()

    plt.show()


###
def distinct_variants_AdvancedEvol(log, outcome = None, time_unit='days', frequency='weekly', case_assignment = 'first_event', type= 'univariate', numeric_agg= 'mean', xtr_outlier_rem = True, cases_initialized = True):
    """Computes and visualizes the time series requested by the identically named DynamicLogPlots instance. 

    Parameters
    ----------
    log : pandas.DataFrame
        Event log
    outcome : str, optional
        Name outcome column in log, by default None
    time_unit : {'microseconds', 'milliseconds', 'seconds', 'minutes', 'hours', 'days', 'weeks'}
        Time unit in which the throughput time of cases is specified, by default 'days'.
    frequency : {'minutely', '5-minutely', '10-minutely', 'half-hourly', 'hourly' '2-hourly', 
                '12-hourly', 'daily', 'weekly', '2-weekly', 'monthly', 'quarterly', 'half-yearly'}
        Frequency by which the observations are grouped together, by default 'weekly'.
    case_assignment : {'first_event', 'last_event', 'max_events'}
        Determines the condition upon which each case is assigned to a certain 
        period, by default 'first_event'.
    type : {'univariate', 'type_tt', 'type_events_case', 'type_outcome'}
        Determines which time series are constructed and visualized, by default 'univariate'.
        For a more detailed explanation of the different 'type' options, see Notes.
    numeric_agg : {'mean', 'median', 'min', 'max', 'std'}
        Determines how periodic quantities are aggregated, by default 'mean'.
    xtr_outlier_rem : bool, optional
        If True, the vertical ranges of the plots are only determined by regular  
        values, i.e. extreme outliers (>q3 + 3*iqr) in the time series are neglected 
        when determining the vertical range, by default True.
    cases_initialized : bool, optional
        _description_, by default True
        
    Notes
    -----
    For a more detailed explanation, see the documentation of the identically named DynamicLogPlots class method. 
    """
    # Inner functions belonging to a specific plot type:
    title_1 = "{} fraction cases belonging to new variants vs. to variants already seen in previous periods".format(frequency)
    label_newprc = "New variants: fraction cases"
    label_oldprc = "Already existing variants: fraction cases"
    ylabel_tt = "{} Throughput Time (in {})".format(numeric_agg, time_unit)
    ylabel_nepc = "{} Number of Events Per Case (NEPC)".format(numeric_agg)
    ylabel_out = "Fraction outcome = True"
    def plt_distVars_uni():
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches([20, 6.25*2])
        # Distinct variants evolution
        plt_period(x, period_df['num_distinct_vars'], ax[0], y_label = "Number Distinct Variants", label = "Number of Distinct Variants ({})".format(frequency), 
                    location = 'left', color = '#1f77b4', title = title_vars)
        ax_0_r= ax[0].twinx()
        # New distinct variants evolution 
        plt_period(x, period_df['num_NEW_distinct_vars'], ax_0_r, y_label= "Number Distinct New Variants", label= "Number of Distinct New Variants ({})".format(frequency),
                location = 'right', color= '#ff7f0e' )
        
        # Periodic fraction of cases belonging to newly introduced variants vs other cases
        plt_period(x, y = period_df["New_prc"], axes = ax[1], y_label = "Fraction cases", number = 1, max_k = 2,
                    title = title_1, label = label_newprc) 
        plt_period(x, y = period_df["Old_prc"], axes = ax[1], y_label = "Fraction cases", number = 2, max_k = 2,
                    title = title_1, label = label_oldprc) 


        if xtr_outlier_rem:
            ax[0].set_ylim(top = max_global_varcounts[0]*1.05)
            ax_0_r.set_ylim(top = max_global_varcounts[1]*1.05)
            max_y = max(max_values_prc[0], max_values_prc[1])
            ax[1].set_ylim(top = max_y*1.05)

        ax[1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
        # st.set_y(1)

    def plt_distVars_tt():
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches([20, 6.25*2])
        title_ttNewOld = "{} {} of TT (in {}) cases belonging to new variants vs. variants already seen in previous periods".format(frequency, numeric_agg, time_unit)
        label_newtt = "New variants' cases: TT"
        label_oldtt = "Already existing variants' cases: TT"
        if cases_initialized:
            title_ci = "Number of Initialized Cases (left) and {} Throughput Time (in {}) (right) over time".format(numeric_agg, time_unit)
            plt_period(x, period_df['total'], ax[0,1], y_label = "# Cases", label = "# cases contained ({})".format(frequency), 
                        location = 'left', color = '#1f77b4', title = title_ci)
            ax_0_r= ax[0,1].twinx()
            #       - periodic numeric_agg of TT
            plt_period(x, period_df[tt_col], ax_0_r, y_label= ylabel_tt, label= "{} Throughput Time ({})".format(numeric_agg, time_unit),
                    location = 'right', color= '#ff7f0e')

            if xtr_outlier_rem:
                ax[0,1].set_ylim(top = max_global_cases*1.05)
                ax_0_r.set_ylim(top = max_global_tt*1.05)
        else: 
            #       - periodic numeric_agg of TT
            title_tt = "{} Throughput Time (in {}) over time".format(numeric_agg, time_unit)
            plt_period(x, period_df[tt_col], ax[0,1], y_label= ylabel_tt, title = title_tt)
            if xtr_outlier_rem:
                ax[0,1].set_ylim(top = max_global_tt*1.05)

        # Distinct count plots:
        plt_period(x, period_df['num_distinct_vars'], ax[0,0], y_label = "Number Distinct Variants", label = "Number of Distinct Variants ({})".format(frequency), 
                    location = 'left', color = '#1f77b4', title = title_vars)
        ax_0_r= ax[0,0].twinx()
        # New distinct variants evolution 
        plt_period(x, period_df['num_NEW_distinct_vars'], ax_0_r, y_label= "Number Distinct New Variants", label= "Number of Distinct New Variants ({})".format(frequency),
                location = 'right', color= '#ff7f0e' )

        # Periodic fraction of cases belonging to newly introduced variants vs other cases
        plt_period(x, y = period_df["New_prc"], axes = ax[1,0], y_label = "Fraction cases", number = 1, max_k = 2,
                    title = title_1, label = label_newprc) 
        plt_period(x, y = period_df["Old_prc"], axes = ax[1,0], y_label = "Fraction cases", number = 2, max_k = 2,
                    title = title_1, label = label_oldprc) 
        # Periodic numeric_agg TT of cases belong to new vs old variants
        plt_period(x, y = period_df["New_tt"], axes = ax[1,1], y_label = ylabel_tt, number = 1, max_k = 2,
                    title = title_ttNewOld, label = label_newtt) 
        plt_period(x, y = period_df["Old_tt"], axes = ax[1,1], y_label = ylabel_tt, number = 2, max_k = 2,
                    title = title_ttNewOld, label = label_oldtt) 


        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_varcounts[0]*1.05)
            ax_0_r.set_ylim(top = max_global_varcounts[1]*1.05)
            # Fractions new vss old vars 
            max_y = max(max_values_prc[0], max_values_prc[1])
            ax[1,0].set_ylim(top = max_y*1.05)
            # TT new vs. old vars
            max_y2 = max(max_values_tt[0], max_values_tt[1])
            ax[1,1].set_ylim(top = max_y2*1.05)


        ax[1,0].set_xlabel("Start dates {} periods".format(frequency))
        ax[1,1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
    
    def plt_distVars_events_case():
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches([20, 6.25*2])
        title_numevNewOld = "{} {} NEPC for cases belonging to new variants vs. variants already seen in previous periods".format(frequency, numeric_agg)
        label_newNumev = "New variants' cases: NEPC"
        label_oldNumev = "Already existing variants' cases: NEPC"
        if cases_initialized:
            title_ci = "Number of Initialized Cases (left) and {} Number of Events Per Case (NEPC) (right) over time".format(numeric_agg)
            plt_period(x, period_df['total'], ax[0,1], y_label = "# Cases", label = "# cases contained ({})".format(frequency), 
                        location = 'left', color = '#1f77b4', title = title_ci)
            ax_0_r= ax[0,1].twinx()
            #       - periodic numeric_agg of NEPC
            plt_period(x, period_df['num_events'], ax_0_r, y_label= ylabel_nepc, label= "{} NEPC".format(numeric_agg),
                    location = 'right', color= '#ff7f0e' )

            if xtr_outlier_rem:
                ax[0,1].set_ylim(top = max_global_cases*1.05)
                ax_0_r.set_ylim(top = max_global_numev*1.05)
        else: 
            #       - periodic numeric_agg of NEPC
            title_nepc = "{} Number of Events Per Case (NEPC) over time".format(numeric_agg)
            plt_period(x, period_df['num_events'], ax[0,1], y_label= ylabel_nepc, title = title_nepc)
            if xtr_outlier_rem:
                ax[0,1].set_ylim(top = max_global_numev*1.05)

        # Distinct count plots:
        plt_period(x, period_df['num_distinct_vars'], ax[0,0], y_label = "Number Distinct Variants", label = "Number of Distinct Variants ({})".format(frequency), 
                    location = 'left', color = '#1f77b4', title = title_vars)
        ax_0_r= ax[0,0].twinx()
        # New distinct variants evolution 
        plt_period(x, period_df['num_NEW_distinct_vars'], ax_0_r, y_label= "Number Distinct New Variants", label= "Number of Distinct New Variants ({})".format(frequency),
                location = 'right', color= '#ff7f0e' )

        # Periodic fraction of cases belonging to newly introduced variants vs other cases
        plt_period(x, y = period_df["New_prc"], axes = ax[1,0], y_label = "Fraction cases", number = 1, max_k = 2,
                    title = title_1, label = label_newprc) 
        plt_period(x, y = period_df["Old_prc"], axes = ax[1,0], y_label = "Fraction cases", number = 2, max_k = 2,
                    title = title_1, label = label_oldprc) 
        # Periodic numeric_agg NEPC of cases belong to new vs old variants
        plt_period(x, y = period_df["New_numev"], axes = ax[1,1], y_label = ylabel_nepc, number = 1, max_k = 2,
                    title = title_numevNewOld, label = label_newNumev) 
        plt_period(x, y = period_df["Old_numev"], axes = ax[1,1], y_label = ylabel_nepc, number = 2, max_k = 2,
                    title = title_numevNewOld, label = label_oldNumev) 

        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_varcounts[0]*1.05)
            ax_0_r.set_ylim(top = max_global_varcounts[1]*1.05)
            # Fractions new vss old vars 
            max_y = max(max_values_prc[0], max_values_prc[1])
            ax[1,0].set_ylim(top = max_y*1.05)
            # NEPC new vs. old vars
            max_y2 = max(max_values_numev[0], max_values_numev[1])
            ax[1,1].set_ylim(top = max_y2*1.05)

        ax[1, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[1, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()
    
    def plt_distVars_outcome():
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches([20, 6.25*2])
        title_outNewOld = "{} fraction of cases with '{}' = True for cases belonging to new variants vs. variants already seen in previous periods".format(frequency, outcome)
        label_newOut = "New variants' cases: fraction outcome = True"
        label_oldOut = "Already existing variants' cases: fraction outcome = True"

        if cases_initialized:
            title_ci = "Number of Initialized Cases (left) and Fraction of cases with '{}' = True (right) over time".format(numeric_agg, outcome)
            plt_period(x, period_df['total'], ax[0,1], y_label = "# Cases", label = "# cases contained ({})".format(frequency), 
                        location = 'left', color = '#1f77b4', title = title_ci)
            ax_0_r= ax[0,1].twinx()
            #       - periodic fraction of cases with outcome = True ( = 1)
            plt_period(x, y= period_df['prc_True'], axes= ax_0_r, y_label= ylabel_out, label= "Fraction outcome '{}' = True".format(outcome),
                    location = 'right', color= '#ff7f0e' )

            if xtr_outlier_rem:
                ax[0,1].set_ylim(top = max_global_cases*1.05)
                ax_0_r.set_ylim(top = max_global_out*1.05)
        else: 
            title_out = "Fraction of cases with '{}' = True over time".format(outcome)
            #       - periodic fraction of cases with outcome = True ( = 1)
            plt_period(x, y= period_df['prc_True'], axes= ax[0,1], y_label= ylabel_out, title = title_out)
            if xtr_outlier_rem:
                ax[0,1].set_ylim(top = max_global_out*1.05)

        # Distinct count plots:
        plt_period(x, period_df['num_distinct_vars'], ax[0, 0], y_label = "Number Distinct Variants", label = "Number of Distinct Variants ({})".format(frequency), 
                    location = 'left', color = '#1f77b4', title = title_vars)
        ax_0_r= ax[0,0].twinx()
        # New distinct variants evolution 
        plt_period(x, period_df['num_NEW_distinct_vars'], ax_0_r, y_label= "Number Distinct New Variants", label= "Number of Distinct New Variants ({})".format(frequency),
                location = 'right', color= '#ff7f0e' )

        # Periodic fraction of cases belonging to newly introduced variants vs other cases
        plt_period(x, y = period_df["New_prc"], axes = ax[1,0], y_label = "Fraction cases", number = 1, max_k = 2,
                    title = title_1, label = label_newprc) 
        plt_period(x, y = period_df["Old_prc"], axes = ax[1,0], y_label = "Fraction cases", number = 2, max_k = 2,
                    title = title_1, label = label_oldprc) 
        # Periodic fraction outcome = True for cases belong to new vs old variants
        plt_period(x, y = period_df["New_prc_True"], axes = ax[1,1], y_label = ylabel_out, number = 1, max_k = 2,
                    title = title_outNewOld, label = label_newOut) 
        plt_period(x, y = period_df["Old_prc_True"], axes = ax[1,1], y_label = ylabel_out, number = 2, max_k = 2,
                    title = title_outNewOld, label = label_oldOut) 

    
        if xtr_outlier_rem:
            ax[0,0].set_ylim(top = max_global_varcounts[0]*1.05)
            ax_0_r.set_ylim(top = max_global_varcounts[1]*1.05)
            # Fractions new vss old vars 
            max_y = max(max_values_prc[0], max_values_prc[1])
            ax[1,0].set_ylim(top = max_y*1.05)
            # Fraction out new vs. old vars
            max_y2 = max(max_values_out[0], max_values_out[1])
            ax[1,1].set_ylim(top = max_y2*1.05)

        ax[1, 0].set_xlabel("Start dates {} periods".format(frequency))
        ax[1, 1].set_xlabel("Start dates {} periods".format(frequency))
        fig.tight_layout()

            
    time_col = determine_time_col(frequency, case_assignment)

    tt_col = determine_tt_col(time_unit)

    local_log = log.copy()
    # Get dataframe containing the case id and variant (as a tuple of activity strings) for each case:
    case_variant = get_variant_case(log)
    local_log = local_log.merge(case_variant, on = 'case:concept:name', how = 'left')

    period_varcounts = get_uniq_varcounts(log = local_log, time_col = time_col)
    if xtr_outlier_rem:
        max_global_varcounts = get_maxrange(period_varcounts[['num_distinct_vars', 'num_NEW_distinct_vars']])
    
    # Get a dataframe with 2 columns: one containing the unique case ids, the other containing "New" if case belongs
    # to a variant that was not encountered until that case's time period (time_col), "Old" otherwise. 
    case_id_log = get_newVar_cases(log = local_log, time_col = time_col)

    local_log = local_log.merge(case_id_log, on = 'case:concept:name', how = 'left') # Added column 'NewOrOld' to local_log

    case_log = local_log.drop_duplicates(subset='case:concept:name').copy()
    # Global periodic counts initialized cases
    period_df = case_log.pivot_table(values= 'case:concept:name',index= time_col, aggfunc='count', fill_value=0)
    period_df.columns = ['total']
    # Periodic fraction of cases belonging to previously (before that period) unseen variants, vs fraction of cases belonging to existing variants.
    period_frac_newOld = case_log.pivot_table(values = 'case:concept:name', index = time_col, columns = 'NewOrOld', aggfunc = 'count', fill_value = 0)
    period_frac_newOld = period_frac_newOld[['New', 'Old']]
    period_frac_newOld.columns = ['New_prc', 'Old_prc']
    period_frac_newOld = period_df[['total']].merge(period_frac_newOld, left_index = True, right_index = True, how = 'left').fillna(0)
    period_frac_newOld = period_frac_newOld[['New_prc', 'Old_prc']].div(period_frac_newOld['total'], axis=0)
    period_df = period_df.merge(period_frac_newOld, left_index = True, right_index = True, how = 'left')

    if xtr_outlier_rem: 
        max_global_cases= get_maxrange(period_df)
        max_global_cases = max_global_cases[0]
        max_values_prc = get_maxrange(period_frac_newOld)

    # Adding periodic numeric_agg num_events
    if type == 'type_events_case':
        # Global periodic NEPC aggregations
        period_numev = case_log.pivot_table(values = 'num_events', index= time_col, aggfunc = numeric_agg, fill_value = 0)
        period_df = period_df.merge(period_numev, left_index=True, right_index=True, how='left')

        # Periodic NEPC aggregations for cases belonging to previously (before that period) unseen variants, vs cases belonging to existing variants. 
        period_numev_newOld = case_log.pivot_table(values = 'num_events', index = time_col, columns = 'NewOrOld', aggfunc = numeric_agg)
        period_numev_newOld = period_numev_newOld[['New', 'Old']]
        period_numev_newOld.columns = ['New_numev', 'Old_numev']
        period_df = period_df.merge(period_numev_newOld, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem: 
            max_global_numev = get_maxrange(period_df[['num_events']])
            max_global_numev = max_global_numev[0]
            max_values_numev = get_maxrange(period_numev_newOld)

    
    # Adding periodic fraction of cases with outcome = True (= 1)
    elif type == 'type_outcome':
        period_outcome = case_log[case_log[outcome]==1].pivot_table("case:concept:name",index= time_col, aggfunc="count", fill_value=0)
        period_outcome.columns = ['num_True']
        period_df = period_df.merge(period_outcome, left_index=True, right_index=True, how='left')
        fillvalues = {'num_True': 0}
        period_df = period_df.fillna(value = fillvalues)
        period_df['prc_True'] = period_df['num_True'] / period_df['total']
        for cat in ['New', 'Old']:
            level_log = case_log[case_log['NewOrOld'] == cat]
            level_prcTrue = get_outcome_percentage(filtered_log= level_log, outcome = outcome, time_col = time_col)
            level_prcTrue.columns = [cat+'_prc_True']
            period_df = period_df.merge(level_prcTrue, left_index= True, right_index= True, how= 'left')

        if xtr_outlier_rem: 
            max_global_out= get_maxrange(period_df[['prc_True']])
            max_global_out= max_global_out[0]
            max_values_out = get_maxrange(period_df[['New_prc_True', 'Old_prc_True']])

    # Adding periodic numeric_agg tt 
    elif type == 'type_tt':
        period_tt= case_log.pivot_table(values= tt_col,index= time_col,aggfunc=numeric_agg, fill_value=0) #column is tt_col
        period_df= period_df.merge(period_tt, left_index=True, right_index=True, how='left')

        # Periodic TT aggregations for cases belonging to previously (before that period) unseen variants, vs cases belonging to existing variants. 
        period_tt_newOld = case_log.pivot_table(values = tt_col, index = time_col, columns = 'NewOrOld', aggfunc = numeric_agg)
        period_tt_newOld = period_tt_newOld[['New', 'Old']]
        period_tt_newOld.columns = ['New_tt', 'Old_tt']
        period_df = period_df.merge(period_tt_newOld, left_index = True, right_index = True, how = 'left')
        if xtr_outlier_rem: 
            max_global_tt = get_maxrange(period_df[[tt_col]])
            max_global_tt = max_global_tt[0]
            max_values_tt = get_maxrange(period_tt_newOld)

    # Adding varcounts 
    period_df = period_df.merge(period_varcounts, left_index = True, right_index = True, how = 'left')

    
    x = period_df.index
    title_vars = "Number of Distinct Variants and New Distinct Variants over time"

    if type == 'univariate':
        plt_distVars_uni()

    elif type == 'type_tt':
        plt_distVars_tt()
    
    elif type == 'type_events_case':
        plt_distVars_events_case()

    elif type == 'type_outcome':
        plt_distVars_outcome()

    plt.show()