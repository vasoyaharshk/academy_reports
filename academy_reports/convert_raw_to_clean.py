import os
import numpy as np
import pandas as pd
import settings
import sys



# get list of paths sorted by date
def path_generator_date(path, pattern):
    paths = {}
    for root, _, file in os.walk(path):
        for f in file:
            if f.endswith(pattern):
                date = f.split('_')[-2]
                if root in paths:
                    paths[root].append((f, date))
                else:
                    paths[root] = [(f, date)]

    for key, values in paths.items():
        values.sort(key=lambda tup: tup[1])
        paths[key] = values

    return paths



# small function to make lists
def make_list(x):
    if x.size <= 1:
        return x
    elif x.isnull().all():
        return np.nan
    else:
        return ','.join([str(x.iloc[i]) for i in range(len(x))])



# raw df to clean df
def transform(df):
    df0 = df
    df0['idx'] = range(1, len(df0) + 1)
    df1 = df0.set_index('idx')
    df2 = df1.pivot_table(index='TRIAL', columns='MSG', values=['START', 'END'],
                          aggfunc=make_list)

    df3 = df1.pivot_table(index='TRIAL', columns='MSG', values='VALUE',
                          aggfunc=lambda x: x if x.size == 1 else x.iloc[0])
    df4 = pd.concat([df2, df3], axis=1, sort=False)

    columns_to_drop = [item for item in df4.columns if type(item) == tuple and (item[1].startswith('_Tup')
                       or item[1].startswith('_Transition') or item[1].startswith('_Global')
                       or item[1].startswith('_Condition'))]
    df4.drop(columns=columns_to_drop, inplace=True)

    columns_to_drop = [item for item in df4.columns if type(item) == str and (item.startswith('_Tup')
                       or item.startswith('_Transition') or item.startswith('_Global')
                       or item.startswith('_Condition'))]
    df4.drop(columns=columns_to_drop, inplace=True)

    df4.columns = [item[1] + '_' + item[0] if type(item) == tuple else item for item in df4.columns]

    df4.replace('', np.nan, inplace=True)
    df4.dropna(subset=['TRIAL_END'], inplace=True)
    df4['trial'] = range(1, len(df4) + 1)

    list_of_columns = df4.columns

    start_list = [item for item in list_of_columns if item.endswith('_START')]
    end_list = [item for item in list_of_columns if item.endswith('_END')]
    other_list = [item for item in list_of_columns if item not in start_list and item not in end_list]

    states_list = []
    for item in start_list:
        states_list.append(item)
        for item2 in end_list:
            if item2.startswith(item[:-5]):
                states_list.append(item2)

    new_list = ['date', 'trial', 'subject', 'task', 'stage', 'checksum', 'box', 'TRIAL_START', 'TRIAL_END']
    new_list += states_list + other_list
    new_list = pd.Series(new_list).drop_duplicates().tolist()

    df4 = df4[new_list]

    return df4




# main loop

def main(arg):

    print(arg)

    paths = path_generator_date(settings.data_directory, 'raw.csv')


    for directory, filenames in paths.items():

        print('parsing directory', directory)

        global_filename = os.path.basename(os.path.normpath(directory)) + '.csv'
        global_path = os.path.join(directory, global_filename)

        df_all = None

        for filename in filenames:

            clean_filename = filename[0][:-8] + '.csv'
            raw_path = os.path.join(directory, filename[0])
            clean_path = os.path.join(directory, clean_filename)

            raw_df = pd.read_csv(raw_path, sep=';')

            if raw_df['TRIAL'].iloc[-1] > 1:

                if arg == ['all']:
                    clean_df = transform(raw_df)
                    clean_df.to_csv(clean_path, sep=';', header=True, index=False)
                else:
                    clean_df = pd.read_csv(clean_path, sep=';')

                if df_all is None:
                    clean_df.insert(loc=0, column='session', value=1)
                    df_all = clean_df
                else:
                    clean_df['session'] = [(int(df_all['session'].iloc[-1]) + 1)] * clean_df.shape[0]
                    df_all = pd.concat([df_all, clean_df], sort=True)
            else:
                pass

        df_all.to_csv(global_path, header=True, index=False, sep=';')



if __name__ == "__main__":
    main(sys.argv[1:])

