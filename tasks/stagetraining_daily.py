import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from academy_reports import utils
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# PLOT COLORS
correct_first_c = 'green'
correct_other_c = 'limegreen'
miss_c = 'black'
incorrect_c = 'red'
punish_c = 'firebrick'
water_c = 'teal'
lines_c = 'gray'
lines2_c = 'silver'
vg_c = 'MidnightBlue'
ds_c = 'RoyalBlue'
dm_c = 'CornflowerBlue'
dl_c = 'LightSteelBlue'

stim_c = 'gold'
correct_th_c = 'green'
repoke_th_c = 'orangered'
label_kwargs = {'fontsize': 9}

def stagetraining_daily (df, save_path, date):

    ##################### PARSE #####################

    ###### RELEVANT VARIABLES ######
    subject = df.subject.iloc[0]
    weight = df.subject_weight.iloc[0]
    total_trials = int(df.trial.iloc[-1])
    valid_trials = (df.loc[df['trial_result'] != 'miss']).shape[0]
    missed_trials = total_trials - valid_trials
    reward_drunk = int(df.reward_drunk.iloc[-1])
    stage = df.stage.iloc[0]
    substage = df.substage.iloc[0]

    # BPODCRASH ERROR
    try:
        df = df.loc[df['BPODCRASH'].isnull()] #remove corrupt columns
        print('WARNING: BPODCRASH ERROR!')
    except:
        pass

    # STIMULUS WIDTH AND THRESHOLDS
    stim_width = df.width.iloc[0] / 2
    try:
        correct_th = df.correct_th.iloc[10] / 2
        repoke_th = df.repoke_th.iloc[10] / 2
    except:
        correct_th = df.correct_th.iloc[0] / 2
        repoke_th = df.repoke_th.iloc[0] / 2
    threshold_lines = [stim_width, correct_th, repoke_th]
    threshold_lines_c = [stim_c, correct_th_c, repoke_th_c]

    # CHANCE
    mask=3

    chance_p = 1 / mask
    chance_lines = []
    for i in range(1, mask + 1, 1):
        chance_lines.append(i * chance_p)
    chance_lines = [chance_lines[0], chance_lines[-1]]

    #BINNING:
    x_positions = df.x.unique().tolist()
    x_positions.sort()
    if mask==3:
        r_positions = [60, 175, 290]
    l_edge = int(min(r_positions) - correct_th)
    r_edge = int(max(r_positions) + correct_th)
    bins_resp = np.linspace(l_edge, r_edge, len(r_positions) + 1)



    #CHECK IF REPOKING IS ALLOWED
    if repoke_th > correct_th:
        repoking_bool = True
    else:
        repoking_bool = False


    ######  RELEVANT COLUMNS  ######
    # add columns (when absent)
    column_list = ['STATE_Correct_first_START', 'STATE_Miss_START', 'STATE_Punish_START', 'STATE_Correct_other_START',
                   'STATE_Incorrect_START', 'STATE_Incorrect_END', 'STATE_Response_window2_START',  'STATE_Response_window2_END',
                   'STATE_Correct_first_reward_START', 'STATE_Correct_other_reward_START', 'STATE_Miss_reward_START']
    for col in column_list:
        if col not in df.columns:
            df[col] = np.nan

    # Column with simplified ttypes (controls considered normals)
    df['trial_type_simple'] = df['trial_type']
    df.loc[df['trial_type'].str.contains('DS'), 'trial_type_simple']= 'DS'
    df.loc[df['trial_type'].str.contains('DM'), 'trial_type_simple'] = 'DM'

    # ttype colors
    df['ttype_colors'] = vg_c
    df.loc[df.trial_type_simple == 'DS', 'ttype_colors'] = ds_c
    df.loc[df.trial_type_simple == 'DM', 'ttype_colors'] = dm_c
    df.loc[df.trial_type_simple == 'DL', 'ttype_colors'] = dl_c

    # tresult colors
    df['treslt_colors'] = miss_c
    df.loc[(df.trial_result == 'correct_first', 'treslt_colors')] = correct_first_c
    df.loc[(df.trial_result == 'correct_other', 'treslt_colors')] = correct_other_c
    df.loc[(df.trial_result == 'incorrect', 'treslt_colors')] = incorrect_c
    df.loc[(df.trial_result == 'punish', 'treslt_colors')] = punish_c


    ###### CONVERT STRINGS TO LISTS ######
    conversion_list = ['STATE_Incorrect_START', 'STATE_Incorrect_END',
                       'STATE_Response_window2_START', 'STATE_Response_window2_END']

    new_conversion_list = []
    for column in conversion_list:
        try:
            df[column].str.contains(',')  # means that contains multiple values
            new_conversion_list.append(column)
        except:  # remove from conversion list
            if column == 'STATE_Incorrect_START':
                new_conversion_list.append(column)

    new_conversion_list.extend(['response_x', 'response_y'])
    df = utils.convert_strings_to_lists(df, new_conversion_list)


    ######  COLUMNS OPERATIONS ######
    # CALCULATE LATENIES
    try: #add nans to empty list, if not error
        df['STATE_Response_window2_END'] = df['STATE_Response_window2_END'].apply( lambda x: [np.nan] if len(x) == 0 else x)
        df['response_window_end'] = df['STATE_Response_window2_END'].apply(lambda x: x[-1])
    except:
        df['response_window_end'] = df['STATE_Response_window2_END'].apply(lambda x: x if type(x) == float else [np.nan])

    df['response_window_end'] = df['response_window_end'].fillna(df['STATE_Response_window_END'])
    df['reward_time'] = df['STATE_Correct_first_reward_START'].fillna(0) + df[
        'STATE_Correct_other_reward_START'].fillna(0) + df['STATE_Miss_reward_START'].fillna(0)
    df['lick_latency'] = df['reward_time'] - df['response_window_end']

    # CALCULATE STIMULUS DURATION &  DELAY
    df['corridor_time'] = df['STATE_Fixation3_END'] - df['STATE_Fixation1_START']
    df['stim_onset'], df['stim_duration'], df['stim_offset'] = zip(
        *df.apply(lambda row: utils.stimulus_duration_calculation(row), axis=1))

    df['delay_total'] = df['response_window_end'] - df['stim_offset']
    df['delay_corridor'] = df['STATE_Fixation3_END'] - df['stim_offset']
    # invalidate delay_corridor in trials with stim_duration lonfer than the end of corridor
    df.loc[df['trial_type'] == 'VG', 'delay_corridor'] = np.nan
    df.loc[((df['trial_type'] == 'DS') & (df['stim_dur_ds'] > 0)), 'delay_corridor'] = np.nan


    ###### CREATE RESPONSES DF ######
    # needed columns before the unnest
    df['responses_time'] = df.apply(lambda row: utils.create_responses_time(row), axis=1)
    df['response_result'] = df.apply(lambda row: utils.create_reponse_result(row), axis=1)
    # unnest
    resp_df = utils.unnesting(df, ['response_x', 'response_y', 'responses_time', 'response_result'])

    ######  RELEVANT COLUMNS  ######
    resp_df['resp_latency'] = resp_df['responses_time'] - resp_df['STATE_Response_window_START']
    resp_df['error_x'] = resp_df['response_x'] - resp_df['x']
    resp_df['abs_error_x'] = resp_df['error_x'].abs()
    resp_df['correct_bool'] = np.where(resp_df['correct_th'] / 2 >= resp_df['error_x'].abs(), 1, 0)
    resp_df['miss_bool'] = np.where(resp_df['trial_result'] =='miss', 1, 0)
    resp_df.loc[(resp_df.trial_result == 'miss', 'correct_bool')] = np.nan
    # Correct_bool column: 1 correct; 0 incorrects/punish; nan miss

    # rresult colors
    resp_df['rreslt_colors'] = miss_c
    resp_df.loc[(resp_df.response_result == 'correct_first', 'rreslt_colors')] = correct_first_c
    resp_df.loc[(resp_df.response_result == 'correct_other', 'rreslt_colors')] = correct_other_c
    resp_df.loc[(resp_df.response_result == 'incorrect', 'rreslt_colors')] = incorrect_c
    resp_df.loc[(resp_df.response_result == 'punish', 'rreslt_colors')] = punish_c

    ###### USEFUL LISTS ######
    ttypes = df.trial_type.unique().tolist()
    ttypes, ttypes_c, = utils.order_lists(ttypes, 'ttypes')  # order trial types list
    ttypes_simple = df.trial_type_simple.unique().tolist()
    ttypes_simple, ttypes_simple_c, = utils.order_lists(ttypes_simple, 'ttypes')  # order ttypes simplified list
    treslts = df.trial_result.unique().tolist()
    treslts, treslts_c = utils.order_lists(treslts, 'treslts')  # order trial results list
    rreslts_c = resp_df.rreslt_colors.unique().tolist() # responses results (extended) list


    ######  CREATE SUBDATAFRAMES  ######
    first_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='first', inplace=False).copy()
    if repoking_bool == True:
        last_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='last', inplace=False).copy()

    ######  TOTAL ACCURACIES CALCULATION  ######
    total_acc_first_poke = int(first_resp_df.correct_bool.mean() * 100)
    if repoking_bool == True:
        total_acc_last_poke = int(last_resp_df.correct_bool.mean() * 100)

    total_acc_dict = {}
    for ttype in ttypes_simple:
        ttype_df =  first_resp_df.loc[first_resp_df['trial_type_simple'] == ttype]
        single_acc = ttype_df.correct_bool.mean() if ttype_df.shape[0] != 0 else 0.0
        total_acc_dict[ttype] = single_acc

    total_acc_ttype = ''
    for key, value in total_acc_dict.items():
        total_acc_ttype = total_acc_ttype + '  /  First Acc ' + str(key) + ': ' + str(int(value * 100)) + "%"

    ##################### PLOT #####################

    ############## PAGE 1 ##############
    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(11.7, 11.7))  # apaisat

        #### HEADER
        s1 = ('Subject: ' + str(subject) +
              '  /  Task: ' + str(df.task.iloc[0]) +
              '  /  Date: ' + str(date) +
              '  /  Box: ' + str(df.box.iloc[0]) +
              '  /  Stage: ' + str(int(stage)) +
              '  /  Substage: ' + str(int(substage)) + '\n')

        s2 = ('Total trials: ' + str(int(total_trials)) +
              '  /  Valids: ' + str(valid_trials) +
              '  /  Missed: ' + str(missed_trials) +
              '  /  Weight: ' + str(weight) + " g" +
              '  /  Rel. weight: ' + str(round(utils.relative_weights(subject, weight), 2)) + "%" +
              '  /  Reward drunk: ' + str(reward_drunk) + " ul" + '\n')

        s3 = ('First Acc global: ' + str(total_acc_first_poke) + '%' + total_acc_ttype + '\n')


        ### PLOT 0:
        # STIMULUS EXTENSION VS TRIAL INDEX (only appear in stage 2 when stimulus extension changes)
        if stage == 2:
            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=4, colspan=39)
            #variables selection
            hlines = [0.4, 0.3, 0.2, 0.1, 0]
            color = ds_c            #substage 1 --> reduce DS stim extension
            var = 'stim_dur_ds'
            if df['task'].str.contains('StageTraining_8B').any():
                pass
            else:
                if substage == 2:       #substage 2 --> reduce DM stim extension
                    color = dm_c
                    var = 'stim_dur_dm'
                elif substage == 3:     #substage 3 --> reduce DM stim extension
                    color = dl_c
                    var = 'stim_dur_dl'

            sns.lineplot(x=first_resp_df.trial, y=first_resp_df[var], ax=axes, color=color) #marker='o'
            axes.hlines(y=hlines, xmin=min(first_resp_df.trial), xmax=max(first_resp_df.trial), color=lines2_c, linestyle=':', linewidth=1)
            axes.set_ylabel('Stim extension (s)', label_kwargs)
            axes.set_xlabel('')
            axes.xaxis.set_ticklabels([])
            axes.set_xlim(1, total_trials + 1)

            subs= first_resp_df.loc[((first_resp_df['trial']>10))]
            try:
                ymin= min(subs[var])
                ymax= max(subs[var])
            except: #low nº of trials
                ymax= min(first_resp_df[var])
                ymin=ymax
            try:
                axes.set_ylim(ymin - 0.1, ymax + 0.1)
            except:
                print('null values plot 0')
                if type(ymin) == str:
                    ymin = 0
                elif type(ymax) == str:
                    ymax = 0.6

           #legend
            label = 'Max: ' + str(round(ymax, 3))+ ' s\n' + \
                    'Min: ' + str(round(ymin, 3))+ ' s'
            axes.text(0.9, 1.3, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                      bbox=dict(facecolor='white', edgecolor=lines2_c, alpha=0.5))

            axes = plt.subplot2grid((50, 50), (5, 0), rowspan=7, colspan=39) #axes for the next plot
            axes.text(0.1, 0.9, s1 + s2 + s3, fontsize=8, transform=plt.gcf().transFigure)  # header


        else: #define header and next plot axes if prev plot was not done
            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=12, colspan=39)  # axes for the next plot
            axes.text(0.1, 0.9, s1 + s2 + s3, fontsize=8, transform=plt.gcf().transFigure) # header



        #### PLOT 1: ACCURACY VS TRIAL INDEX
        # we use the previous defined axes
        ttype_palette = sns.set_palette(ttypes_simple_c, n_colors=len(ttypes_simple)) # set the palette
        labels = list(ttypes_simple)
        colors = list(ttypes_simple_c)

        ### trial type accuracies######
        for ttype, ttype_df in first_resp_df.groupby('trial_type_simple'):
            ttype_color = ttype_df.ttype_colors.iloc[0]
            ttype_df['acc'] = utils.compute_window(ttype_df.correct_bool, 20)
            sns.lineplot(x=ttype_df.trial, y=ttype_df.acc, ax=axes, color=ttype_color, marker='o', markersize=5)

        if repoking_bool == True: # separate in first and last poke
            for ttype, ttype_df in last_resp_df.groupby('trial_type_simple'):
                ttype_color = ttype_df.ttype_colors.iloc[0]
                ttype_df['acc'] = utils.compute_window(ttype_df.correct_bool, 20)
                sns.lineplot(x=ttype_df.trial, y=ttype_df.acc, ax=axes, color=ttype_color, marker='o', linestyle='--', markersize=5)

            linestyle = len(colors) * ['-']
            labels.extend(['First poke', 'Last poke'])
            colors.extend(['black', 'black'])
            linestyle.extend(['-', 'dotted'])
            label = 'First acc: ' + str(total_acc_first_poke) + '%' + '\n' + 'Last acc: ' + str(
                total_acc_last_poke) + '%'
            axes.text(0.9, 1.1, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                      bbox=dict(facecolor='white', edgecolor=lines2_c, alpha=0.5))
        else: # global accuracy
            first_resp_df['acc'] = utils.compute_window(first_resp_df.correct_bool, 20)
            sns.lineplot(x=first_resp_df.trial, y=first_resp_df.acc, ax=axes, color='black')
            labels.append('All')
            colors.append('black')
            linestyle = len(colors) * ['-']

        # axis
        axes.hlines(y=chance_lines, xmin=0, xmax=total_trials, color=lines_c, linestyle=':', linewidth=1)
        axes.fill_between(df.trial, chance_lines[0], 0, facecolor=lines2_c, alpha=0.2)
        axes.set_xlabel('Trial', label_kwargs)
        axes.set_xlim([1, total_trials + 1])
        utils.axes_pcent(axes, label_kwargs)
        try:
            axes.get_legend().remove()
        except:
            pass


        #### PLOT 2: ACCURACY VS TRIAL TYPE
        axes = plt.subplot2grid((50, 50), (0, 40), rowspan=12, colspan=10)
        x_min = -0.5
        x_max = len(ttypes_simple) -0.5

        if repoking_bool == True:  # add last poke
            #sns.pointplot(x=last_resp_df.trial_type, y=last_resp_df.correct_bool, ax=axes, ci=68, color='black', linestyles=["--"])  #linestyles error

            sns.pointplot(x=last_resp_df.trial_type, y=last_resp_df.correct_bool, ax=axes, ci=68, color='black')
        else:
            grouped_df = first_resp_df.groupby('trial_type').agg({'correct_bool': 'mean', 'trial_type_simple': max}).reset_index()
            sns.stripplot(x='trial_type_simple', y='correct_bool', data=grouped_df, order=ttypes_simple, hue="trial_type",
                      jitter=False, size=4, palette='Greys') #split controls
            try:
                sns.legend.remove()
            except:
                pass
        sns.pointplot(x='trial_type_simple', y='correct_bool', data= first_resp_df, order=ttypes_simple,
                      markersize=10, ci=68, color='black')

        # axis
        axes.hlines(y=chance_lines, xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
        axes.fill_between(np.linspace(x_min, x_max, 2), chance_p, 0, facecolor=lines2_c, alpha=0.2)
        axes.set_xlabel('', label_kwargs)
        utils.axes_pcent(axes, label_kwargs)
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        try:
            axes.set_xticklabels(ttypes, rotation=40)
        except:
            pass

        # legend
        lines = [Line2D([0], [0], color=colors[i], marker='o', markersize=7, markerfacecolor=colors[i], linestyle=linestyle[i]) for i in
                 range(len(colors))]
        axes.legend(lines, labels, title='Trial type', fontsize=8, loc='center', bbox_to_anchor=(1, 1.1))


        #### PLOT 3: ACCURACY VS STIMULUS POSITION
        axes = plt.subplot2grid((50, 50), (15, 0), rowspan=11, colspan=15)
        x_min = 0
        x_max = 400 #screen size

        sns.lineplot(x='x', y="correct_bool", data=first_resp_df, hue='trial_type_simple', hue_order=ttypes_simple,
                    marker='o', markersize=8, err_style="bars", ci=68, ax=axes)

        axes.hlines(y=chance_lines, xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
        axes.fill_between(np.arange(x_min, x_max, 1), chance_p, 0, facecolor=lines2_c, alpha=0.2)
        try:
            axes.get_legend().remove()
        except:
            pass

        # axis
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])
        utils.axes_pcent(axes, label_kwargs)


        ### PLOT 4: ERRORS VS STIMULUS POSITION
        axes = plt.subplot2grid((50, 50), (26, 0), rowspan=11, colspan=15)

        sns.lineplot(x='x', y="error_x", data=first_resp_df, hue='trial_type_simple',hue_order=ttypes_simple,
                     marker='o', markersize=8, err_style="bars", ci=68, ax=axes)

        axes.hlines(y=[-correct_th / 2, correct_th / 2], xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
        axes.set_xlabel('Stimulus \ position\ (x_{t})\ (mm)%', label_kwargs)
        axes.set_ylabel('Error (mm)', label_kwargs)
        if len(first_resp_df.trial_type.unique()) > 1:
            axes.get_legend().remove()


        ### PLOT 5: RESPONSE COUNTS
        axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=15)

        for ttype, ttype_df in first_resp_df.groupby('trial_type_simple'):
            ttype_color = ttype_df.ttype_colors.iloc[0]
            hist, bins = np.histogram(ttype_df.response_x, bins=bins_resp, density=True)
            try:
                sns.lineplot(x=r_positions, y=hist, marker='o', markersize=8, err_style="bars", color=ttype_color)
            except:
                pass
        axes.set_xlim(x_min, x_max)
        axes.set_xlabel('$Responses\ (r_{t})\ (mm)$', label_kwargs)
        axes.set_ylabel('Prob normalized', label_kwargs)


        ### PLOT 6: RESPONSE COUNTS SORTED BY STIMULUS POSITION
        side_colors = ['lightseagreen', 'bisque', 'orange']
        axes_loc = [17, 28, 39]
        colspan = 10

        # if mask != len(x_positions):  # less than 3 choices
        #     side_colors = []
        #     for x in r_positions:
        #         if x< 100: # Left choice
        #             side_colors.append('lightseagreen')
        #         elif x >100 and x < 300: # Central choice
        #             side_colors.append('bisque')
        #         elif x > 300:  # Right choice
        #             side_colors.append('orange')

        side_palette = sns.set_palette(side_colors, n_colors=len(side_colors))  # palette creation
        first_resp_df['rt_bins'] = pd.cut(first_resp_df.response_x, bins=bins_resp, labels=r_positions,
                                          include_lowest=True)

        for idx in range(len(x_positions)):
            axes = plt.subplot2grid((50, 50), (16, axes_loc[idx]), rowspan=12, colspan=colspan)
            subset = first_resp_df.loc[first_resp_df['x'] == x_positions[idx]]
            axes.set_title('x_{t}\ :%' + str(x_positions[idx]), fontsize=13, fontweight='bold')
            sns.countplot(subset.rt_bins, ax=axes, palette=side_colors)
            axes.set_xlabel('')
            if idx != 0:
                axes.set_ylabel('')
                axes.yaxis.set_ticklabels([])
                if idx == int((len(x_positions) / 2)):
                    axes.set_xlabel('$Responses\ (r_{t})\ (mm)$')

        # legend
        labels = list(r_positions)
        lines = [Patch(facecolor=c, edgecolor=c) for c in side_colors]
        axes.legend(lines, labels, fontsize=8, title= 'Rt', loc='center', bbox_to_anchor=(1.1, 1))

        ### PLOT 7: RESPONSE LATENCIES   # we look to all the reponses time
        axes = plt.subplot2grid((50, 50), (31, 18), rowspan=25, colspan=15)

        if stage == 1:
            y_max = 30
        else:
            y_max = 10
        sns.boxplot(x='x', y='resp_latency', hue='rt_bins', data=first_resp_df, color='white', linewidth=0.5,
                        showfliers=False, ax=axes)
        resp_df['rt_bins'] = pd.cut(resp_df.response_x, bins=bins_resp, labels=r_positions,
                                          include_lowest=True)
        sns.stripplot(x='x', y='resp_latency', hue='rt_bins', data=resp_df, dodge=True, ax=axes)
        axes.set_ylabel("Response latency (sec)", label_kwargs)
        axes.set_xlabel('Stimulus \ position\ (x_{t})\ (mm)%', label_kwargs)
        axes.set_ylim(0, y_max)
        axes.get_legend().remove()

        ### PLOT 8: LICKPORT LATENCIES   # we look only the trial time
        axes = plt.subplot2grid((50, 50), (31, 38), rowspan=25, colspan=13)

        y_max = 20
        sns.boxplot(x='trial_result', y='lick_latency', data=df, color='white', linewidth=0.5, showfliers=False,
                    ax=axes, order=treslts)
        sns.stripplot(x="trial_result", y="lick_latency", color=water_c, data=df, dodge=True,
                      ax=axes, order=treslts)

        axes.set_xticklabels(treslts, fontsize=9, rotation=35)
        axes.set_ylabel("Lickport latency (sec)", label_kwargs)
        axes.set_xlabel("")
        axes.set_ylim(0, y_max)
        sns.despine()


        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()

        ############## PAGE 2 ##############

        plt.figure(figsize=(11.7, 11.7))  # apaisat

        #align to the reponse window onset
        states_list = ['STATE_Correct_first_START', 'STATE_Correct_other_START','STATE_Miss_START',
                       'STATE_Punish_START', 'reward_time', 'responses_time']
        for idx, state in enumerate(states_list):
            resp_df[state] = resp_df[state] - resp_df['STATE_Response_window_START']

        df['stim_onset_align'] = df['stim_onset'] - df['STATE_Response_window_START']
        df['stim_offset_align'] = df['stim_offset'] - df['STATE_Response_window_START']

        # RASTER PLOT
        x_min = -2
        if stage == 1:
            x_max = 30
        else:
            x_max = 10

        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=42, colspan=25)
        treslt_palette = sns.set_palette(rreslts_c, n_colors=len(rreslts_c))

        sns.scatterplot(x=resp_df.reward_time, y=resp_df.trial, color=water_c, s=20, ax=axes, label='water')
        sns.scatterplot(x=resp_df.responses_time, y=resp_df.trial, hue=resp_df.response_result,
                        style=resp_df.trial_type_simple, s=20, ax=axes)
        # horizontal lines
        axes.hlines(y=df.trial, xmin=x_min, xmax=x_max, color=lines2_c, alpha=0.2, zorder=10).set_linewidth(5)  # horizontal bars each trial
        axes.hlines(y=df.trial, xmin=df.stim_onset_align, xmax=df.stim_offset_align,
                    color=lines_c, alpha=0.2, zorder=10).set_linewidth(5)
        axes.axvline(x=0, color=lines_c, linewidth=1.5, zorder=10)

        axes.set_ylim(-1, total_trials + 1)
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])
        axes.set_ylabel('Trials', label_kwargs)
        axes.set_xlim(x_min, x_max)
        axes.get_legend().remove()

        # HISTOGRAM OF LATENCIES
        axes = plt.subplot2grid((50, 50), (43, 0), rowspan=7, colspan=25)
        size_bins = 0.4
        bins = np.arange(0, x_max, size_bins)
        sns.distplot(resp_df.reward_time, kde=False, bins=bins, color=water_c, ax=axes,
                     hist_kws={'alpha': 0.8, 'histtype': 'step', 'linewidth': 2})
        for respres, respres_df in resp_df.groupby('response_result'):
            colors = respres_df.rreslt_colors.iloc[0]
            sns.distplot(respres_df.responses_time, kde=False, bins=bins, color=colors, ax=axes,
                         hist_kws={'alpha': 0.8, 'histtype': 'step', 'linewidth': 2})

        axes.axvline(x=0, color=lines_c, linewidth=1.5, zorder=10)
        axes.set_xlim(x_min, x_max)
        axes.set_xlabel('Latency (sec)', label_kwargs)
        axes.set_ylabel('Number of pokes', label_kwargs)

        # ERRORS TRIAL INDEX
        axes = plt.subplot2grid((50, 50), (0, 26), rowspan=42, colspan=25)
        x_min = -400
        x_max = 400
        sns.scatterplot(x=resp_df.error_x, y=resp_df.trial, hue=resp_df.response_result, style=resp_df.trial_type_simple,
                        s=20, ax=axes, zorder=20)
        axes.hlines(y=df.trial, xmin=x_min, xmax=x_max, color=lines2_c, alpha=0.2, zorder=10).set_linewidth(5)  # horizontal bars each trial

        #vertical lines
        axes.axvspan(-stim_width, stim_width, color=stim_c, alpha=0.2)
        for idx, line in enumerate(threshold_lines):
            axes.axvline(x=line, color=threshold_lines_c[idx], linestyle=':', linewidth=1)
            axes.axvline(x=-line, color=threshold_lines_c[idx], linestyle=':', linewidth=1)

        axes.set_xlabel('')
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        axes.xaxis.set_ticklabels([])
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(-1, total_trials + 1)
        axes.legend(loc='center', bbox_to_anchor=(1, 1), title='Trial result').set_zorder(10)
        sns.despine()

        # ERRORS HISTOGRAM
        #axes = plt.subplot2grid((50, 50), (43, 26), rowspan=7, colspan=25)
        #bins = np.linspace(-r_edge, r_edge, mask * 2 + 6)
        #sns.distplot(first_resp_df.error_x, kde=False, bins=bins, color=lines_c, ax=axes,
        #             hist_kws={'alpha': 0.9, 'histtype': 'step', 'linewidth': 2})



        axes = plt.subplot2grid((50, 50), (43, 26), rowspan=7, colspan=25)
        bins = np.linspace(0, 400, 800)
        sns.histplot(resp_df.response_x, kde=False, bins=bins, color=lines_c, ax=axes)



        # vertical lines
        #axes.axvspan(-stim_width, stim_width, color=stim_c, alpha=0.2)
        #for idx, line in enumerate(threshold_lines):
        #    axes.axvline(x=line, color=threshold_lines_c[idx], linestyle=':', linewidth=1)
        #    axes.axvline(x=-line, color=threshold_lines_c[idx], linestyle=':', linewidth=1)

        #axes.set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
        #axes.set_ylabel('')
        #axes.yaxis.set_ticklabels([])
        #axes.set_xlim(x_min, x_max)

        axes.set_ylim(0, 20)
        sns.despine()


        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()


        ############## PAGE 3 ##############

        plt.figure(figsize=(11.7, 11.7))  # apaisat

        ### PLOT 1: STIMULUS POSITION ACCURACY BY TRIAL TYPE

        df['response_first'] = df['response_x'].str[0]
        df['response_last'] = df['response_x'].str[-1]

        treslt_palette = sns.set_palette(df.treslt_colors.unique(), n_colors=len(df.treslt_colors.unique()))

        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=15, colspan=50)
        sns.scatterplot(x=df.trial, y=df.x, color='white', edgecolor='black', s=10, ax=axes)
        sns.scatterplot(x=df.trial, y=df.response_first, hue=df.trial_result, s=30, ax=axes)

        blocks = [i for i in range(20, total_trials + 1, 20)]
        for line in blocks:
            axes.axvline(line, color=lines2_c, linestyle=':')
        axes.set_ylabel('Touchscreen position (mm)')
        axes.set_xlabel('Trials')
        axes.set_ylim(0, 405)

        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()

        ############## PAGE 4 ##############    LAST PAGE ONLY IF OPTO ON

        if df['task'].str.contains('StageTraining_8B_V2').any():
            df['opto_bool'] = df['opto_bool'].astype(int)
            if 1 in df['opto_bool'].unique():

                plt.figure(figsize=(11.7, 11.7))  # apaisat

                ### PLOT 1: STIMULUS POSITION ACCURACY BY TRIAL TYPE
                x_min = 0
                x_max = 400  # screen size
                opto_colors=['gray', 'gold']
                opto_order=[0, 1]
                y_pos = [0, 17, 35]

                first_resp_df.loc[first_resp_df['y']==1000, 'trial_type_simple']='SIL'

                for idx, ttype in enumerate(['VG', 'DS', 'SIL']):
                    axes = plt.subplot2grid((50, 50), (0, y_pos[idx]), rowspan=11, colspan=15)
                    subset= first_resp_df.loc[first_resp_df['trial_type_simple']==ttype]
                    axes.set_title(ttype, fontsize=13, fontweight='bold')
                    sns.lineplot(x='x', y='correct_bool', data=subset,  hue='opto_bool', hue_order=opto_order, marker='o',
                                 markersize=8, err_style="bars", ci=68, palette=opto_colors)
                    axes.hlines(y=chance_lines, xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
                    axes.fill_between(np.arange(x_min, x_max, 1), chance_p, 0, facecolor=lines2_c, alpha=0.2)

                    # axis
                    axes.set_xlim(x_min, x_max)
                    axes.set_xlabel('Stimulus position (mm)', label_kwargs)
                    utils.axes_pcent(axes, label_kwargs)
                    if idx==0:
                        axes.set_ylabel('Accuracy', label_kwargs)
                        axes.legend(loc='center', bbox_to_anchor=(0.15, 0.15), title='Laser').set_zorder(10)
                    else:
                        axes.set_ylabel('')
                        try:
                            axes.get_legend().remove()
                        except:
                            pass

                ### HEADINGS: LABELING TYPE
                labeling = utils.labeling_class(df.subject.iloc[0])
                axes.text(0.1, 0.9, 'OTPGENETICS SESSION DETAILS    4OHT Labeling: ' + labeling +'\n', fontsize=8, transform=plt.gcf().transFigure)  # header

                ### PLOT 2: LASER ON/OFF
                axes = plt.subplot2grid((50, 50), (16, 0), rowspan=12, colspan=6)
                first_resp_df['count']=1
                counts = first_resp_df.groupby('opto_bool')['count'].sum().reset_index()
                sns.barplot(x='opto_bool', y='count', data=counts, palette=opto_colors)
                axes.set_xlabel('Laser', label_kwargs)
                axes.set_ylabel('Nº of trials', label_kwargs)

                ### PLOT 3: % MISSES WITH LIGHT
                axes = plt.subplot2grid((50, 50), (16, 10), rowspan=12, colspan=6)
                df['miss_bool'] = np.where(df['trial_result'] == 'miss', 1, 0)
                counts = df.drop_duplicates(subset=['trial'], keep='first', inplace=False).copy()
                counts = counts.groupby('opto_bool')['miss_bool'].sum().reset_index()
                sns.barplot(x='opto_bool', y='miss_bool', data=counts, palette=opto_colors)
                axes.set_xlabel('Laser', label_kwargs)
                axes.set_ylabel('Nº of misses', label_kwargs)


                ### PLOT 4: RESPONSE LATENCIES WITH LIGHT
                axes = plt.subplot2grid((50, 50), (16, 20), rowspan=12, colspan=15)
                to_plot= first_resp_df.loc[first_resp_df['trial']>8]
                sns.stripplot(x='trial_result', y='resp_latency', order=['correct_first', 'punish'], hue='opto_bool', hue_order=opto_order, data=to_plot,
                              palette=opto_colors, dodge=True, ax=axes)
                sns.boxplot(x='trial_result', y='resp_latency',  order=['correct_first', 'punish'], hue='opto_bool', hue_order=opto_order, data=to_plot,
                            color='white', linewidth=0.5, showfliers=False, ax=axes)
                axes.set_ylabel("Response latency (sec)", label_kwargs)
                axes.set_xlabel('Trial Outcome', label_kwargs)
                axes.set_xticklabels(['Correct', 'Incorrect'])
                axes.get_legend().remove()
                axes.set_ylim(0, 10)


                # SAVING AND CLOSING PAGE
                sns.despine()
                pdf.savefig()
                plt.close()

        print('New daily report completed successfully')