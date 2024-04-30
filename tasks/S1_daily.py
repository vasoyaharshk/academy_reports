import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from datetime import timedelta, datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# PLOT COLORS
correct_c = 'green'
miss_c = 'black'
punish_c = 'firebrick'
left_c = 'teal'
right_c = 'orange'
water_c = 'blue'
lines_c = 'silver'
label_kwargs = {'fontsize': 9}


def stagetraining_daily (df, save_path, date):

    ##################### PARSE #####################

    ###### RELEVANT VARIABLES ######
    subject = df.subject.iloc[0]
    total_trials = int(df.trial.iloc[-1])
    valid_trials = (df.loc[df['trial_result'] != 'miss']).shape[0]
    missed_trials = total_trials - valid_trials
    stage = df.stage.iloc[0]
    substage = df.substage.iloc[0]
    reward_drunk = int(df.reward_drunk.iloc[-1])


    df['correct_bool'] = np.where(df['trial_result'] == 'correct', 1, 0)
    df['miss_bool'] = np.where(df['trial_result'] == 'miss', 1, 0)
    df['side_bool'] = np.where(df['side'] == 'right', 1, 0)
    df['lick_latency'] = df.STATE_Response_window_END - df.STATE_Response_window_START

    total_acc = int(df.correct_bool.mean() * 100)
    lick_latency_mean = df.lick_latency.mean()
    chance=0.5

    

    ##################### PLOTS #####################

    ############## PAGE 1 ##############
    with PdfPages(save_path) as pdf:
        plt.figure(figsize=(8.3, 11.7)) 
        
        
        axes = plt.subplot2grid((50, 50), (0, 1), rowspan=5, colspan=50)
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
              '  /  Reward drunk: ' + str(reward_drunk) + " ul" + '\n')

        s3 = ('Acc global: ' + str(total_acc) + '%'+ '\n')

        axes.text(0.1, 0.9, s1 + s2 +S3, fontsize=8, transform=plt.gcf().transFigure)


        ### PLOT 1: SCATTER OF LICKS
        sns.scatterplot(x=df.trial, y=df.side_bool, hue=df.trial_result, hue_order=['correct', 'miss'], palette=[correct_c, miss_c], s=30, ax=axes)
        axes.legend(fontsize=6, loc='center', bbox_to_anchor=(0.95, 1.25))
        axes.axis('off')


        ### PLOT 2: LICK LATENCY BY TRIAL INDEX
        axes = plt.subplot2grid((50, 50), (6, 0), rowspan=10, colspan=50)
        sns.scatterplot(x=df.trial, y=df.lick_latency, color=water_c, s=30, ax=axes)
        sns.lineplot(x=df.trial, y=df.lick_latency, color=water_c, ax=axes)
        axes.set_ylabel('Lick latency (sec)')

        label = 'Mean: ' + str(round(lick_latency_mean, 1)) + ' sec'
        axes.text(0.85, 1.2, label, transform=axes.transAxes, fontsize=8, verticalalignment='top')


        ### PLOT 3: ACCURACY BY TRIAL INDEX
        axes = plt.subplot2grid((50, 50), (18, 0), rowspan=10, colspan=50)

        sns.despine()

        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()


