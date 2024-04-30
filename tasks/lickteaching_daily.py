import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# PLOT COLORS
correct_c = 'green'
miss_c = 'black'
water_c = 'teal'
lines_c = 'gray'


def lickteaching_daily (df, save_path, date):

    # RELEVANT COLUMNS
    df['colors'] = miss_c
    df.loc[(df.trial_result == 'correct_first', 'colors')] = correct_c
    df['lick_latency'] = df.STATE_Wait_for_reward_END - df.STATE_Wait_for_reward_START

    # RELEVANT VARIABLES
    total_trials = int(df.trial.iloc[-1])
    valid_trials = (df.loc[df['trial_result'] == 'correct_first']).shape[0]
    missed_trials = total_trials - valid_trials
    reward_drunk = int(df.reward_drunk.iloc[-1])
    lick_latency_mean = df.lick_latency.mean()

    # PAGE 1:

    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(8.3, 11.7))  # A4 vertical

        # HEADER
        axes = plt.subplot2grid((50, 50), (0, 1), rowspan=4, colspan=50)
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
        df['y_val']=1
        colors = df.colors.unique().tolist()
        custom_palette = sns.set_palette(sns.color_palette(colors))
        # labels = df.trial_result.unique().tolist()
        # lines = [Line2D([0], [0], color=c, marker='o', markersize=4, markerfacecolor=c) for c in colors]

        sns.scatterplot(x=df.trial, y=df.y_val, hue=df.trial_result, palette=custom_palette, s=30, ax=axes)
        axes.axis('off')
        axes.legend(fontsize=6, loc='center', bbox_to_anchor=(0.95, 1.25))

        # LICK LATENCY PLOT
        axes = plt.subplot2grid((50, 50), (5, 0), rowspan=8, colspan=50)
        sns.scatterplot(x=df.trial, y=df.lick_latency, color=water_c, s=30, ax=axes)
        sns.lineplot(x=df.trial, y=df.lick_latency, color=water_c, ax=axes)
        axes.hlines(y=5, xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        axes.set_ylabel('Lick latency (sec)')

        label = 'Mean: ' +str(round(lick_latency_mean, 1)) + ' sec'
        axes.text(0.85, 1.2, label, transform=axes.transAxes, fontsize=8, verticalalignment='top')
        sns.despine()


         # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()
