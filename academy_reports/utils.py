import os
import numpy as np
import pandas as pd

if 'SLACK_BOT_TOKEN' in os.environ:
    import slack
else:
    print(os.environ)

# GET LIST OF PATHS SORTED BY NAME
def path_generator(path, pattern):
    paths = []
    for root, _, file in os.walk(path):
        for f in file:
            if f.endswith(pattern):
                paths.append(os.path.join(root, f))
    return sorted(paths)


# CONVERT STRING COLUMNS OF DF TO LISTS
def convert_strings_to_lists(df, columns):
    """
    If the csv contains a column that is ',' separated, that column is read as a string.
    We want to convert that string to a list of values. We try to make the list float or string.
    """
    def tolist(stringvalue):
        if isinstance(stringvalue, str):
            try:
                stringvalue = stringvalue.split(sep=',')
                try:
                    val = np.array(stringvalue, dtype=float)
                except:
                    val = np.array(stringvalue)
            except:
                val = np.array([])
        elif np.isnan(stringvalue):
            return np.array([])
        else:
            val = np.array([stringvalue])
        return val.tolist()

    for column in columns:
        df[column] = df[column].apply(tolist)
    return df


# UNNESTING LISTS IN COLUMNS DATAFRAMES
def unnesting(df, explode):
    """
    Unnest columns that contain list creating a new row for each element in the list.
    The number of elements must be the same for all the columns, row by row.
    """
    print(df)
    length = df[explode[0]].str.len()
    idx = df.index.repeat(length)
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    finaldf = df1.join(df.drop(explode, 1), how='left')
    finaldf.reset_index(drop=True, inplace=True)

    length2 = [list(range(l)) for l in length]
    length2 = [item + 1 for sublist in length2 for item in sublist]
    name = explode[0] + '_index'
    finaldf[name] = length2

    return finaldf

# SUBJECT TAGS
def subjects_tags():
    '''Identifies the subject depending on the tag
     ECOHAB reads tags with reversed order by pairs'''
    all_subjects = ['man', 'T1', 'T2', 'T3',
                    'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A50', 'A51', 'A52']
    all_mv_tags = ['041A9DB979', '041A9C89B3', '041A9C7958', '0419A8212D',
                   '0417CA5FDE', '041A9DBD90', '0419A86ECB', '0419A8218D', '0417CA97FA', '0419A8701C',
                   '041A9D7BE0', '0419A822D2', '041A9DBDF9', '041A9DB349', '0419A81BFB', '041A9D86C5']
    all_colors = ['lightsteelblue', 'mediumseagreen', 'greenyellow', 'salmon',
              'yellow', 'orange', 'tomato', 'crimson', 'mediumvioletred',
              'darkorchid', 'darkblue', 'royalblue', 'lightskyblue', 'mediumaquamarine',
              'green', 'yellowgreen']

    all_ecohab_tags = []  # ECOHAB reads tags with reversed order by pairs
    for tag in all_mv_tags:  # loop thought MV tags
        tag_r = tag[::-1]  # revert
        new_tag = ''
        for (front, back) in zip(tag_r[0::2], tag_r[1::2]):  # invert 2 by 2
            new_tag += back + front
        all_ecohab_tags.append(new_tag)

    return all_subjects, all_ecohab_tags, all_colors

# BASAL WEIGHTS
def relative_weights(subject, weight):
    basal_weights = {
    'A5': '32.68', 'A6': '31.46', 'A7': '30.40', 'A8': '31.38', 'A9': '31.65', 'A10': '27.71', 'A11': '31.20', 'A12': '27.72',
    'MA1': '31.3', 'MA2': '25.9', 'MA3': '28.2', 'MA4': '27', 'MA5': '30.9',
    'A13':'23.4', 'A14':'21.63', 'A15':'21.8', 'A16':'21.87', 'A17':'22.7', 'A18':'21.37', 'A19':'23.7', 'A20':'24.1',
    'MA6': '24.84', 'MA7': '26.48', 'MA8': '27.51', 'MA9': '24', 'MA10': '25',
    'MA11': '24.84', 'MA12': '26.48', 'MA13': '27.51', 'MA14': '24', 'MA15': '25',
    'A21':'19.77', 'A22':'20.1', 'A23':'21.1', 'A24':'22.73', 'A25':'21.3','A26':'20.4', 'A27':'21.8','A28':'22.77', 'A29':'22.8', 'A30':'24.1',
    'A31':'21.9', 'A32':'22', 'A33':'22.1', 'A34':'26.6', 'A35':'22.5','A36':'23.2', 'A37':'21.7','A38':'22.3', 'A39':'22.6', 'A40':'21.6',
    'A41':'22.6', 'A42':'27.6', 'A43':'23.2', 'A44':'22.2', 'A45':'25.2','A46':'21.4', 'A47':'24.9','A48':'25.6', 'A49':'22.7', 'A50':'23.9',
    'A51':'23.5', 'A52':'27.2',
    'A53':'23.4', 'A54':'24.1', 'A55':'23.7','A56':'22.4', 'A57':'23.9','A58':'23.7', 'A59':'21.9', 'A60':'27.5',
    'A61':'23.9', 'A62':'26.6', 'A63':'28.4',
    'A64': '26.9', 'A65': '24.5', 'A66': '26.5', 'A67': '26.2', 'A68': '27.6',
    'A69': '21.4', 'A70': '21.9', 'A71': '23.8', 'A72': '21.7', 'A73': '24.2',
    'A74': '24.6', 'A75': '26.0', 'A76': '26.0', 'A77': '24.3',
    'A78':'27.9', 'A79':'26.9', 'A80':'26.1', 'A81':'25.1', 'A82':'27.2',
    'A83': '26.65', 'A84':'26.3', 'A85':'25.9', 'A86':'25.75', 'A87':'25.6', 'A88':'24.3', 'A89':'25.8', 'A90': '24.3',
    'A91':'24.7', 'A92':'24.7', 'A93':'22.4', 'A94':'22.5', 'A95':'22.5', 'A96':'22.5', 'A97':'21.5',
    'B1': '26.6', 'B2': '23.3', 'B3': '26.7', 'B4': '26.6', 'B5': '30.6', 'B6': '29.6', 'B7': '25.2', 'B8': '27.5',
    'B9': '28.3', 'B10': '28.0', 'B11': '25.7', 'B12': '24.4', 'B13': '25.8', 'B14': '26',
    'T1': '33' , 'T2':'32.7','T3': '32.6' , 'T4':'32.9', 'T5':'26.7','T6': '27.5' , 'T7':'27.7', 'T8':'22.6', 'T9':'30.9'}

    for key, value in basal_weights.items():
        if subject == key:
            basal_weight_subj = float(value)
            relative_weight_subj = weight / basal_weight_subj * 100
            return relative_weight_subj

# COMPUTE WINDOW AVERAGE
def compute_window(data, runningwindow):
    """
    Computes a rolling average with a length of runningwindow samples.
    """
    performance = []
    for i in range(len(data)):
        if i < runningwindow:
            performance.append(round(np.mean(data[0:i + 1]), 2))
        else:
            performance.append(round(np.mean(data[i - runningwindow:i]), 2))
    return performance


# COLLECT ALL REPONSES TIMES IN A COLUMN
def create_responses_time(row):
    try:
        result = row['STATE_Incorrect_START'].tolist().copy()
    except (TypeError, AttributeError):
        result = row['STATE_Incorrect_START'].copy()
    items = [row['STATE_Correct_first_START'], row['STATE_Correct_other_START'], row['STATE_Punish_START']]
    for item in items:
        if not np.isnan(item):
            result += [item]
    return result


# RESPONSE RESULT COLUMN
def create_reponse_result(row):
    result = ['incorrect'] * len(row['STATE_Incorrect_START'])
    if row['trial_result'] != 'miss' and row['trial_result'] != 'incorrect':
        result += [row['trial_result']]
    return result


# CREATE CSVS
def create_csv(df, path):
    df.to_csv(path, sep=';', na_rep='nan', index=False)


# PECRCENTAGE AXES
def axes_pcent(axes, label_kwargs):
    """
    convert y axis form 0-1 to 0-100%
    """
    axes.set_ylabel('Accuracy (%)', label_kwargs)
    axes.set_ylim(0, 1.1)
    axes.set_yticks(np.arange(0, 1.1, 0.1))
    axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])

# CHANCE CALCULATION
def chance_calculation(correct_th):
    """
    calculates chance baseline considering the number of possible responses
    """
    screen_size = 1440 * 0.28
    chance = correct_th*2 / screen_size
    return chance


# ORDER LISTS
def order_lists(list, type):
    '''Returns ordered lists with the differnt trial types and its corresponding colors lists'''
    vg_c = 'MidnightBlue'
    ds_c = 'RoyalBlue'
    dm_c = 'CornflowerBlue'
    dl_c = 'LightSteelBlue'
    if type == 'ttypes':
        order = ['VG', 'DS', 'DSc1', 'DSc2', 'DM', 'DMc1', 'DL']
        c_order = [vg_c, ds_c, ds_c, ds_c, dm_c, dm_c, dl_c]
    elif type == 'treslts':
        order = ['correct_first', 'correct_other', 'punish', 'incorrect', 'miss']
        c_order = ['green', 'limegreen', 'firebrick', 'red', 'black']
    elif type == 'probs':
        order = ['pvg', 'pds', 'pdsc1', 'pdsc2', 'pdm', 'pdmc1', 'pdl']
        c_order = [vg_c, ds_c, ds_c, ds_c, dm_c, dm_c, dl_c]

    ordered_list = []
    ordered_c_list = []

    for idx, i in enumerate(order):
        if i in list:
            ordered_list.append(i)
            ordered_c_list.append(c_order[idx])
    return ordered_list, ordered_c_list

# STATS AFTER GROUPBY FOR REPEATING BIAS CALC
def stats_function(df, groupby_list):
    """Creates stats_ dataframe with the groupby rows desired and CRB calculated"""
    stats_ = df.groupby(groupby_list).agg({'version': 'max', 'chance': 'max', 'correct_bool': 'mean',
                                               'rep_bool': ['mean', 'std', 'sum', 'count']}).reset_index()
    stats_.columns = list(map(''.join, stats_.columns.values))
    stats_['norm'] = stats_['rep_boolmean'] / stats_['chancemax']  # normalize by chance
    stats_['CRB'] = stats_['norm'] - 1  # corrected repeating bias calculation
    return stats_

# STIMULUS CALCULATION
def stimulus_duration_calculation(row):
    ''' Calculates the stimulus onset, offset and duration.
        Extends stimulus duration adding extra time up to the maximum when necessary '''
    if 'DS' in row['trial_type']:
        if row['trial_type'] == 'DS':
            stim_onset = row['STATE_Fixation1_START']
        elif row['trial_type'] == 'DSc1':
            stim_onset = row['STATE_Fixation3_START']
        elif row['trial_type'] == 'DSc2':
            stim_onset = row['STATE_Fixation2_START']

        stim_offset = row['STATE_Fixation3_END']
        stim_duration = stim_offset - stim_onset

        if row['stim_dur_ds'] > 0:  # stimulus duration extended to the next state
            stim_dur_ext = stim_duration + row['stim_dur_ds']
            max_dur = row['response_window_end'] - stim_onset
            if stim_dur_ext <= max_dur:  # extend when don't overcome max
                stim_duration = stim_dur_ext
            elif stim_dur_ext > max_dur:  # take the maximum when overcome
                stim_duration = max_dur
            stim_offset = stim_onset + stim_duration  # correct stimulus offset

    elif 'DM' in row['trial_type']:
        if row['trial_type'] == 'DM':
            stim_onset = row['STATE_Fixation1_START']
        elif row['trial_type'] == 'DMc1':
            stim_onset = row['STATE_Fixation2_START']

        stim_offset = row['STATE_Fixation2_END']
        stim_duration = stim_offset - stim_onset

        if row['stim_dur_dm'] > 0:  # stimulus duration extended to the next state
            stim_dur_ext = stim_duration + row['stim_dur_dm']
            max_dur = row['STATE_Fixation3_END'] - stim_onset
            if stim_dur_ext <= max_dur:  # extend when don't overcome max
                stim_duration = stim_dur_ext
            elif stim_dur_ext > max_dur:  # take the maximum when overcome
                stim_duration = max_dur
            stim_offset = stim_onset + stim_duration  # correct stimulus offset

    elif 'DL' in row['trial_type']:
        stim_onset = row['STATE_Fixation1_START']
        stim_offset = row['STATE_Fixation1_END']
        stim_duration = stim_offset - stim_onset

        if row['stim_dur_dl'] > 0:  # stimulus duration extended to the next state
            stim_dur_ext = stim_duration + row['stim_dur_dl']
            max_dur = row['STATE_Fixation2_END'] - stim_onset
            if stim_dur_ext <= max_dur:  # extend when don't overcome max
                stim_duration = stim_dur_ext
            elif stim_dur_ext > max_dur:  # take the maximum when overcome
                stim_duration = max_dur
            stim_offset = stim_onset + stim_duration  # correct stimulus offset

    elif 'VG' in row['trial_type']:
        stim_onset = row['STATE_Fixation1_START']
        stim_offset = row['response_window_end']
        stim_duration = stim_offset - stim_onset

    try:
        return stim_onset, stim_duration, stim_offset
    except:
        return np.nan, np.nan, np.nan



# LABELING 4OHT CLASSIFICATION
def labeling_class(subject):
    ''' Categorize type of labeling for each subject'''
    central_ch = ['A61', 'A62', 'A64', 'A65', 'A69', 'A75', 'A60', 'A80']
    homecage= ['A66', 'A70', 'A54', 'A71', 'A56']
    if subject in central_ch:
        return 'Central_ch'
    elif subject in homecage:
        return 'homecage'
    else:
        return 'no labeling'


if 'SLACK_BOT_TOKEN' in os.environ:
    def slack_spam(msg='hey buddy', filepath=None, userid='U8J8YA66S'):
        """this sends msgs through the bot,
        avoid spamming too much else it will get banned/timed-out"""
        ids_dic = {
            'jordi': 'U8J8YA66S',
        }

        if (userid[0]!='U') and (userid[0]!='#'): # asumes it is a first name
            try:
                userid = ids_dic[userid.lower()]
            except:
                raise ValueError('double-check slack channel id (receiver)')

        token = os.environ.get('SLACK_BOT_TOKEN')
        if token is None:
            print('no SLACK_BOT_TOKEN in environ')
            raise EnvironmentError('no SLACK_BOT_TOKEN in environ')
        else:
            try:
                client = slack.WebClient(token=token)
                if (os.path.exists(filepath)) and (filepath is not None):
                    response = client.files_upload(
                            channels=userid,
                            file=filepath,
                            initial_comment=msg)
                elif filepath is None:
                    response = client.chat_postMessage(
                        channel=userid,
                        text=msg)
            except Exception as e:
                print(e) # perhaps prints are caught by pybpod
