from academy_reports import utils
from academy_reports import settings
from academy_reports import arg
import os
import pandas as pd
from datetime import timedelta, datetime
from tasks.lickteaching_daily import lickteaching_daily
from tasks.touchteaching_daily import touchteaching_daily
from tasks.stagetraining_daily import stagetraining_daily
from tasks.intersession import intersession
from tasks.ecohab_report import ecohab_report
import warnings
warnings.filterwarnings('ignore')


# MAIN
def main():

    ########################################## ECOHAB REPORTS ##########################################

    # CONVERT RAW TO CLEAN CSV
    # print('-------------------------------------------')
    # print('Generating clean ecohab csvs')
    # raw_paths = utils.path_generator(settings.data_directory2, 'raw.csv')
    # if not os.path.exists(settings.data_directory2):
    #     os.makedirs(settings.data_directory2)
    #
    # for path in raw_paths:  # loop by different raw csv
    #     clean_path = path[:-7] + "clean.csv"
    #     print(clean_path)
    #
    #     if not os.path.exists(clean_path):  # if clean not done
    #         df = pd.read_csv(path, sep='\t')
    #
    #         if df.empty:
    #             continue
    #
    #         ##################### PARSE #####################
    #         ### ECOHAB detects weird RDIF lectures sometimes: good lectures finish in 04 and are consistently detected
    #         all_subjects, all_ecohab_tags, all_colors = utils.subjects_tags()
    #
    #         ### Remove aberrant detections
    #         match_tags = []
    #         df['tag'] = df['RFID_detected'].apply(lambda x: x.strip("0"))  # remove final 0
    #
    #         for x in df.tag.unique():  # loop thoug all tags detected
    #             n_coincidences = []
    #             for tag in all_ecohab_tags:  # loop thought correct tags
    #                 if x in tag:  # check if parts of the incorrect tag coincide with real tags
    #                     n_coincidences.append(x)
    #             if len(
    #                     n_coincidences) == 1:  # check if tag detected coincides with more than one real tag, we only want 1 coincidence
    #                 match_tags.append(x)
    #
    #         unique_match_tags = []  # list with the matching tags
    #         for x in match_tags:
    #             if x not in unique_match_tags:
    #                 unique_match_tags.append(x)
    #
    #         df = df.loc[((df['tag'].isin(match_tags)))]  # remove aberrant detections
    #
    #         ### Create a column containing the corrected tags
    #         def tag_correction(x):
    #             if len(x) < 10:
    #                 for tag in all_ecohab_tags:  # loop thought correct tags
    #                     if x in tag:
    #                         return tag
    #             else:
    #                 return x
    #
    #         df['tag'] = df['tag'].apply(lambda x: tag_correction(x))
    #
    #         ### add subject names column
    #         df['subject'] = df['tag'].replace(all_ecohab_tags, all_subjects)
    #         subjects = df.subject.unique()
    #         subjects.sort()  # subjects list sorted
    #
    #         ### Create a column with a colr assigned to each subject
    #         df['colors'] = df['subject'].replace(all_subjects, all_colors)
    #
    #         ### Datetime column
    #         df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    #
    #         ### Generate clean dataframe
    #         subjects_df = None
    #         for s in subjects:  # loop by subject
    #             df_s = df.loc[df.subject == s]
    #             df_s['duration'] = timedelta(seconds=0)
    #             clean_row_list = []  # create empty list
    #             first = True
    #
    #             for index, row in df_s.iterrows():
    #                 if first:  # skip first one
    #                     previous_row = row
    #                     first = False
    #                 else:
    #                     if (row['Antena_number'] == previous_row['Antena_number'] and row['Datetime'] - previous_row[
    #                         'Datetime'] < timedelta(seconds=2)) | index == df.index[-1]: # same event
    #                         row['duration'] = row['Datetime'] - previous_row['Datetime'] + previous_row[
    #                             'duration']  # save event time
    #                         previous_row = row
    #                     else:  # another event (save the previous)
    #                         clean_row_list.append(previous_row)
    #                         previous_row = row
    #
    #             df_s_clean = pd.DataFrame(clean_row_list)  # convert rows to df
    #
    #             df_s_clean['prev_Datetime'] = df_s_clean.Datetime.shift()
    #             df_s_clean['next_Datetime'] = df_s_clean.Datetime.shift(-1)
    #             df_s_clean['prev_antena'] = df_s_clean.Antena_number.shift()
    #             df_s_clean['next_antena'] = df_s_clean.Antena_number.shift(-1)
    #
    #             ### Box inference
    #             df_s_clean.loc[(df_s_clean.prev_antena == 2) & (df_s_clean.Antena_number == 1), 'box'] = 'A'
    #             df_s_clean.loc[(df_s_clean.prev_antena == 7) & (df_s_clean.Antena_number == 8), 'box'] = 'A'
    #             df_s_clean.loc[(df_s_clean.prev_antena == 5) & (df_s_clean.Antena_number == 6), 'box'] = 'B'
    #             df_s_clean.loc[(df_s_clean.prev_antena == 8) & (df_s_clean.Antena_number == 7), 'box'] = 'B'
    #             df_s_clean.loc[(df_s_clean.prev_antena == 3) & (df_s_clean.Antena_number == 4), 'box'] = 'C'
    #             df_s_clean.loc[(df_s_clean.prev_antena == 6) & (df_s_clean.Antena_number == 5), 'box'] = 'C'
    #             df_s_clean.loc[(df_s_clean.prev_antena == 1) & (df_s_clean.Antena_number == 2), 'box'] = 'D'
    #             df_s_clean.loc[(df_s_clean.prev_antena == 4) & (df_s_clean.Antena_number == 3), 'box'] = 'D'
    #
    #             df_s_clean.fillna(method='ffill', inplace=True)
    #             df_s_clean['box_time'] = df_s_clean.next_Datetime - df_s.Datetime
    #
    #             if subjects_df is None:
    #                 subjects_df = df_s_clean
    #             else:
    #                 subjects_df = pd.concat([subjects_df, df_s_clean])
    #
    #         ### save clean df
    #         subjects_df.to_csv(clean_path, sep=';')
    #
    # # MERGE CLEAN CSVS
    # clean_paths = utils.path_generator(settings.data_directory2, 'clean.csv')
    # dfs = []
    # for path in clean_paths:  # loop by different clean csvs
    #     df = pd.read_csv(path, sep=';')
    #     dfs.append(df)
    #
    #     global_ecoh_df = pd.concat(dfs)
    #     utils.create_csv(global_ecoh_df, settings.data_directory2 + '/global_ecoh.csv')
    #
    # # MAKE THE REPORT
    # print('Making Ecohab report')
    # date_s = global_ecoh_df.Date.iloc[0]
    # date_e = global_ecoh_df.Date.iloc[-1]
    # print('From ' + str(date_s))
    # print('To ' + str(date_e))
    # file_name = 'Ecohab_report_' + date_s.replace('.', '') + '_' + date_e.replace('.', '') + '.pdf'
    # save_path = os.path.join(settings.save_directory2, file_name)
    # ecohab_report(global_ecoh_df, save_path)

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

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

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

                save_directory = os.path.join(settings.save_directory, subject)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                # INTERSESSIONS
                try:
                    file_name_intersesion = subject + '_intersession.pdf'
                    save_path_intersesion = os.path.join(save_directory, file_name_intersesion)
                    intersession(df.copy(), save_path_intersesion)
                except:
                     print('Error performing the intersession')
                     pass

                # DAILY REPORTS
                for sess, session in df.groupby('session'):
                    subject = session.subject.iloc[0]
                    task = session.task.iloc[0]
                    stage = session.stage.iloc[0]
                    try:
                        date = datetime.fromtimestamp(session.STATE_Start_task_START.iloc[0]).strftime("%Y%m%d-%H%M%S")
                        print(date)
                    except:
                        bad_date=session.STATE_Start_task_START.iloc[0]
                        print(bad_date)
                    print(task[0:13])
                    print(sess)

                    file_name = subject + '_' + task + '-' + str(stage) + '_' + date + '.pdf'
                    save_path = os.path.join(save_directory, file_name)

                    print(save_path)

                    if not os.path.exists(save_path): #ONLY DONE IF NOT EXISTS
                            if task == 'LickTeaching':
                                print("daily 1")
                                lickteaching_daily(session.copy(), save_path, date)
                            elif task == 'TouchTeaching':
                                touchteaching_daily(session.copy(), save_path, date)
                            elif task[0:13] == 'StageTraining':
                                stagetraining_daily(session.copy(), save_path, date)
                            elif task == 'S1':
                                S1_daily(session.copy(), save_path, date)
                            else:
                                print('Task not found for file:', path, 'task:', task)

                    else:
                        print('Already done!')

        # GLOBAL DF
        print('Generating global df')
        global_df = pd.concat(dfs)
        save_directory = os.path.join(settings.save_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        utils.create_csv(global_df, save_directory + '/global_trials.csv')

        global_df_pi = global_df[global_df['task'].str.contains('Probability', case=False, na=False)].copy()
        utils.create_csv(global_df_pi, save_directory + '/global_trials_pi.csv')

        global_df_st = global_df[global_df['task'].str.contains('StageTraining', case=False, na=False)].copy()
        utils.create_csv(global_df_st, save_directory + '/global_trials_st.csv')
        print('END!')


# MAIN
if __name__ == "__main__":
    main()

