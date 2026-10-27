from academy_reports import utils
from academy_reports import settings
from academy_reports import arg
import os
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from datetime import timedelta, datetime
from tasks.lickteaching_daily import lickteaching_daily
from tasks.touchteaching_daily import touchteaching_daily
from tasks.stagetraining_daily import stagetraining_daily
from tasks.intersession import intersession
from tasks.ecohab_report import ecohab_report
import warnings
warnings.filterwarnings('ignore')
import re



# MAIN
def main():

    #################################### DAILY & INTERSESSIONS ####################################
    print('')
    print('Generating dailies')

    try:
        path = arg.file[0]
        file_name = os.path.basename(path)
        file_name = file_name.split(".")[0]

        df = pd.read_csv(path, sep=';')

        subject = df.subject.iloc[0]
        task = df.task.iloc[0]
        date = datetime.fromtimestamp(df.STATE_Start_task_START.iloc[0]).strftime("%Y%m%d-%H%M%S")

        save_directory = os.path.join(settings.save_directory_manual, subject)

        # if not os.path.exists(save_directory):
        #     os.makedirs(save_directory)

        file_name = file_name + '.pdf'
        save_path = os.path.join(save_directory, file_name)

        if task == 'LickTeaching':
            lickteaching_daily(df, save_path, date)
        elif task == 'TouchTeaching':
            touchteaching_daily(df, save_path, date)
        elif task[0:13] == 'StageTraining':
            stagetraining_daily(df, save_path, date)
        else:
            print('Task not found for file:', path, 'task:', task)


    except:
        raw_paths = utils.path_generator(settings.data_directory, '.csv')

        if not os.path.exists(settings.data_directory):
            os.makedirs(settings.data_directory)

        dfs = []
        for path in raw_paths:

            #sort, only analyze general csvs
            subject = os.path.basename(path)
            print(subject)
            if len(subject) <= 15 and subject != 'session.csv':
                df = pd.read_csv(path, sep=';')
                dfs.append(df)

                subject, ext = os.path.splitext(subject)
                print('')
                print('Starting report '+str(subject))

        # GLOBAL DF
        print('Generating global df')
        global_df = pd.concat(dfs)

        # Remove all the blank trials:
        global_df = global_df.loc[
            ~(
                    ((global_df['trial_length'] == 0.1) & (global_df['trial_result'].isna())) |
                    (global_df['subject'].isin(['m2', 'm3']))
            )
        ].copy()

        save_directory = os.path.join(settings.save_directory)
        # if not os.path.exists(save_directory):
        #     os.makedirs(save_directory)
        # utils.create_csv(global_df, save_directory + '/global_trials.csv')

        # Delete all files in the save_directory (as string)
        for filename in os.listdir(save_directory):
            file_path = os.path.join(save_directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # File paths
        global_trials_pi_file = os.path.join(save_directory, 'global_trials_pi_rv2.csv')
        global_trials_wm_file = os.path.join(save_directory, 'global_trials_wm_rv2.csv')

        # Save the full global dataframe
        # utils.create_csv(global_df, global_trials_file)

        # Filter for tasks that contain 'Probability'
        global_df_pi = global_df[global_df['task'].str.contains(r'Probability|Cognitive_Bias', case=False, na=False)].copy()
        # global_df_pi = global_df[global_df['task'].str.contains('Probability', case=False, na=False)].copy()
        utils.create_csv(global_df_pi, global_trials_pi_file)

        # Filter for tasks that do NOT contain 'Probability': Working Memory
        global_df_wm = global_df[~global_df['task'].str.contains('Probability', case=False, na=False)].copy()
        #utils.create_csv(global_df_wm, save_directory + '/global_trials_wm.csv')

        # Filter for early training tasks
        # ---------------------------------------------------------------
        # build the list of “early-training” tasks automatically
        # (i.e. every task whose name does NOT contain “Probability”
        #  or “Cognitive_Bias”, case-insensitive)
        # ---------------------------------------------------------------
        exclude = re.compile(r'probability|cognitive_bias', re.I)
        early_training_tasks = [
            t for t in global_df['task'].dropna().unique()
            if not exclude.search(t)
        ]
        global_df_early = global_df[
            global_df['task'].str.contains('|'.join(early_training_tasks), case=False, na=False)].copy()
        utils.create_csv(global_df_early, os.path.join(save_directory, 'early_training_rv2.csv'))

        print('END!')



# MAIN
if __name__ == "__main__":
    main()

