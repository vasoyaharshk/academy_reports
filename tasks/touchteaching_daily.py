import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# PLOT COLORS
correct_first_c = 'green'
miss_c = 'black'
water_c = 'teal'
lines_c = 'gray'
wmdl_c = '#a55194'

# BINNING
bins_resp=np.linspace(0, 400, 4)


def touchteaching_daily (df, save_path, date):

    # REMOVE LIST BRACKETS
    # df['response_x'] = df['response_x'].apply(lambda x: x.replace('[', '').replace(']', ''))
    df["response_x"] = pd.to_numeric(df.response_x, errors='coerce')

    # RELEVANT COLUMNS
    df['trial_result'] = 'miss'
    df['colors'] = miss_c
    df.loc[(df.STATE_Correct_first_START > 0, 'trial_result')] = 'correct'
    df.loc[(df.STATE_Correct_first_START > 0, 'colors')] = correct_first_c
    df['resp_latency'] = df.STATE_Response_window_END - df.STATE_Response_window_START

    if set(['STATE_Miss_reward_START']).issubset(df.columns):
        df['lick_time'] = df.STATE_Correct_first_reward_START.fillna(0) + df.STATE_Miss_reward_START.fillna(0)
    else:
        df['lick_time'] = df.STATE_Correct_first_reward_START
    df['lick_latency'] = df.lick_time - df.STATE_Response_window_END


    # RELEVANT VARIABLES
    total_trials = int(df.trial.iloc[-1])
    valid_trials = (df.loc[df['trial_result'] == 'correct']).shape[0]
    missed_trials = total_trials - valid_trials
    reward_drunk =  valid_trials * 8
    response_x_mean = df.response_x.mean()
    resp_latency_mean = df.resp_latency.mean()
    lick_latency_mean = df.lick_latency.mean()


    # PAGE 1:
    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(8.3, 11.7))  # A4 vertical

        # HEADER
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=8, colspan=50)
        s1 = ('Subject name: ' + str(df.subject.iloc[0]) +
              '  /  Task: ' + str(df.task.iloc[0]) +
              '  /  Date: ' + str(date) +
              '  /  Box: ' + str(df.box.iloc[0]) +
              '  /  Weight: ' + str(df.subject_weight.iloc[0]) + " g" + '\n')

        s2 = ('Total trials: ' + str(int(total_trials)) +
              '  /  Valid trials: ' + str(valid_trials) +
              '  /  Missed trials: ' + str(missed_trials) +
              '  /  Reward drunk: ' + str(reward_drunk) + " ul"+ '\n')

        axes.text(0.1, 0.9, s1+s2, fontsize=8, transform=plt.gcf().transFigure)

        # TRIAL RESULT
        colors = df.colors.unique().tolist()
        custom_palette = sns.set_palette(sns.color_palette(colors))

        df['response_x_plot'] = df['response_x'].replace(np.nan, 200)
        sns.scatterplot(x=df.trial, y=df.response_x_plot, hue=df.trial_result, palette=custom_palette, s=30, ax=axes)
        axes.hlines(y=[200], xmin=0, xmax=total_trials, color='gray', linestyle=':')
        #axes.set_ylabel('$Responses\ (r_{t})\ (mm)%$')
        axes.set_ylabel(r'$Responses\ (r_{t})\ (mm)\%$')
        axes.set_xlabel('')
        axes.set_ylim(0, 410)
        axes.set_xlim(0, total_trials+1)
        axes.legend(fontsize=6, loc='center', bbox_to_anchor=(0.95, 1.25))

        label = 'Mean: ' + str(round(response_x_mean, 1)) + ' mm'
        axes.text(0.85, 0.9, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.5))

        # Response latency plot
        axes = plt.subplot2grid((50, 50), (10, 0), rowspan=8, colspan=50)
        sns.scatterplot(x=df.trial, y=df.resp_latency, color=wmdl_c, s=30, ax=axes)
        sns.lineplot(x=df.trial, y=df.resp_latency, color=wmdl_c, ax=axes)
        axes.hlines(y=5, xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        axes.set_ylabel('Response latency (sec)')
        axes.set_ylim(0, 60)
        axes.set_xlim(0, total_trials+1)
        axes.set_xlabel('')

        label = 'Mean: ' + str(round(resp_latency_mean, 1)) + ' sec'
        axes.text(0.85, 0.9, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.5))

        # Lick latency plot
        axes = plt.subplot2grid((50, 50), (20, 0), rowspan=8, colspan=50)
        sns.scatterplot(x=df.trial, y=df.lick_latency, color=water_c, s=30, ax=axes)
        sns.lineplot(x=df.trial, y=df.lick_latency, color=water_c, ax=axes)
        axes.hlines(y=5, xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        axes.set_ylabel('Lick latency (sec)')
        axes.set_ylim(0, 60)
        axes.set_xlabel('Trials')
        axes.set_xlim(0, total_trials+1)

        label = 'Mean: ' + str(round(lick_latency_mean, 1)) + ' sec'
        axes.text(0.85, 0.9, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.5))

        # Hist responses
        axes = plt.subplot2grid((50, 50), (30, 0), rowspan=12, colspan=25)
        sns.distplot(df.response_x, kde=False, bins=bins_resp, color=wmdl_c, ax=axes, hist_kws={'alpha':0.9})
        #axes.set_xlabel('$Responses\ (r_{t})\ (mm)%$')
        axes.set_xlabel(r'$Responses\ (r_{t})\ (mm)\%$')
        axes.set_ylabel('Nº of touches')
        sns.despine()


        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()
