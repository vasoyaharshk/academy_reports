import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from academy_reports import utils
from datetime import timedelta, datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from academy_reports import settings


def ecohab_report (df, save_path):

    ################# RELEVANT VARIABLES ################
    subjects = df.subject.unique()
    subjects.sort() #subjects list sorted
    n_subjects = len(subjects)
    n_days= len(df.Date.unique())
    boxes = df.box.dropna().unique().tolist()
    boxes.sort()
    n_boxes=len(boxes)

    # plot colors and parameters
    box_palette = ['cornflowerblue', 'orange', 'seagreen', 'purple']
    antenna_palette = ['cornflowerblue', 'gold', 'orange', 'yellowgreen', 'seagreen',
                       'mediumorchid', 'purple', 'lightskyblue']
    label_kwargs = {'fontsize': 9}

    # Datetime conversions
    df['Datetime'] = df['Datetime'].apply(
        lambda x: np.nan if pd.isnull(x) else datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df['next_Datetime'] = df['next_Datetime'].apply(
        lambda x: np.nan if pd.isnull(x) else datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df['box_time'] = pd.to_timedelta(df['box_time'])

    #################### PLOT #####################

    ############## PAGE 1 ##############
    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(11.7, 15))

        #### HEADER
        s1 = ('Subjects: ' + str(n_subjects) +
              '  /  Days: ' + str(n_days) +
              '  /  Cages: ' + str(n_boxes) +
              '  /  Last time: ' + str(df['Datetime'].max()) + '\n')

        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=1, colspan=35)
        axes.axis('off')
        # axes
        axes.text(0.1, 0.9, s1, fontsize=8, transform=plt.gcf().transFigure)  # header

        # ### PLOT 1: Box raster plot last 5 days
        # select plot width depending on the number of subjects
        last5_df = df.loc[df['Datetime']>= df['Datetime'].max() - timedelta(days=4)]
        row_w= int(30/n_subjects)
        y = 2

        # vertical lines
        days_at_8 = []
        days_at_20 = []
        for date in last5_df['Date'].unique():
            day_datetime = datetime.strptime(date + ':08:00:00.000', '%Y.%m.%d:%H:%M:%S.%f')
            night_datetime = datetime.strptime(date + ':20:00:00.000', '%Y.%m.%d:%H:%M:%S.%f')
            days_at_8.append(day_datetime)
            days_at_20.append(night_datetime)
        x_min = last5_df['Datetime'].min() - timedelta(hours=1)
        x_max = last5_df['Datetime'].max() + timedelta(hours=1)

        #plot
        for n, subject in enumerate(subjects):  # loop thoug differnt subjects
            axes = plt.subplot2grid((50, 50), (y, 0), rowspan=row_w, colspan=50)
            for j, box in enumerate(boxes):  # loop thoug differnt boxes
                subset = last5_df.loc[((last5_df['subject'] == subject) & (last5_df['box'] == box))]
                axes.hlines(y=subset.box, xmin=subset.Datetime, xmax=subset.next_Datetime,
                               color=box_palette[j]).set_linewidth(5)
                for lines in range(len(days_at_8) - 1):
                    axes.axvspan(days_at_20[lines], days_at_8[lines + 1], facecolor='lightgray', zorder=0)

                #axes
                axes.set_ylim(-1, 4)
                axes.set_ylabel(subject)
                axes.set_yticks([])
                axes.set_xlim(x_min, x_max)
                if n != 0:
                    if n < len(last5_df.subject.unique()) - 1:
                        axes.get_xaxis().set_visible(False)
                else:
                    colors = ['orange', 'seagreen', 'cornflowerblue', 'purple']
                    labels = ['B (House)', 'C (Food)', 'A (Food-MV)', 'D (House)']
                    lines = [Patch(facecolor=c, edgecolor=c) for c in colors]
                    axes.legend(lines, labels, title='Box', loc='center', bbox_to_anchor=[0.9, 1.3], ncol=2, fontsize=8,
                                title_fontsize=10, shadow=True, fancybox=True)
                    axes.get_xaxis().set_visible(False)
            y=y+2

        ### PLOT 2: Box occupancy histograms
        # calculate time (hours) by subject box and date
        group_df = df.groupby(['subject', 'box', 'Date']).agg({'box_time': 'sum', 'colors': 'max'}).reset_index()
        group_df['sec'] = round(group_df['box_time'].dt.total_seconds())  # sec values
        group_df['hours'] = group_df['sec'] / (60 * 60)  # hours values

        y = [28, 28, 28, 28,
             36, 36, 36, 36,
             44, 44, 44, 44]
        x = [0, 8, 16, 24,
             0, 8, 16, 24,
             0, 8, 16, 24]

        for i, subject in enumerate(subjects):
            try:
                axes = plt.subplot2grid((50, 50), (y[i], x[i]), rowspan=6, colspan=6)
                subset = group_df.loc[group_df['subject'] == subject]
                sns.barplot(data=subset.sort_values(by=['box']), x="box", y="hours", ax=axes, ci=68,
                            palette=sns.set_palette(sns.color_palette(box_palette)))
                axes.set_title(subject, fontsize=10, fontweight='bold', y=1.0, pad=-2)
                if i== 0 or  i== 4 or i== 8:
                    axes.set_ylabel('Time (hours)')
                else:
                    axes.set_ylabel('')
                if i<8:
                    axes.set_xlabel('')
            except:
                pass

        ### PLOT 3: Antenas histogram
        axes = plt.subplot2grid((50, 50), (28, 35), rowspan=9, colspan=12)
        sns.countplot(x='Antena_number', data=df,  ax=axes,
                      palette=sns.set_palette(sns.color_palette(antenna_palette)))
        axes.set_xlabel('Antena')

        ### PLOT 4: EVENTS LAST HOUR
        axes = plt.subplot2grid((50, 50), (40, 35), rowspan=9, colspan=12)
        subset = df.groupby('subject').last().reset_index()
        sns.stripplot(x='box', y='subject', data=subset, ax=axes, order=['A', 'B', 'C', 'D'],
                      palette=sns.set_palette(sns.color_palette(box_palette)))
        axes.set_xlabel('Box')

        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()


        ############## PAGE 2 ##############
        plt.figure(figsize=(11.7, 15))


        ### PLOT 5:
        row_w = int(30 / n_subjects)
        y = 0

        # sns.set(style="whitegrid")
        # paper_rc = {'lines.markersize': 2}
        # sns.set_context("paper", rc=paper_rc)

        # plot
        for n, subject in enumerate(subjects):  # loop thoug differnt subjects
            axes = plt.subplot2grid((50, 50), (y, 0), rowspan=row_w, colspan=50)
            subset = df.loc[df['subject'] == subject]
            group_df = subset.groupby(['box', 'Date']).agg({'box_time': 'sum', 'colors': 'max'}).reset_index()
            group_df['sec'] = round(group_df['box_time'].dt.total_seconds())  # sec values
            group_df['hours'] = group_df['sec'] / (60 * 60)  # hours values
            # sns.pointplot(x='Date', y='hours', hue='box', hue_order=['A', 'B', 'C', 'D'], data=group_df,
            #               palette=sns.set_palette(sns.color_palette(box_palette)))
            sns.lineplot(x='Date', y='hours', hue='box', hue_order=['A', 'B', 'C', 'D'], data=group_df,
                         marker='o', markersize=5, err_style="bars", ci=68, ax=axes, palette=sns.set_palette(sns.color_palette(box_palette)))
            # axes
            axes.set_ylim(0, 15)
            axes.set_ylabel(subject)
            axes.set_yticks([])
            axes.get_xaxis().set_visible(False)
            axes.get_legend().remove()
            y = y + 2

        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()








