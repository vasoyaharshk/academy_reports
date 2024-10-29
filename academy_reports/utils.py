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
    length = df[explode[0]].str.len()
    idx = df.index.repeat(length)
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx
    finaldf = df1.join(df.drop(columns=explode), how='left')
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
    all_subjects = ['ciri', 'gal', 'joy', 'luna', 'sorrel', 'sparky']
    all_mv_tags = ['041AFD256C', '041AFD5192', '041AFA3C87', '041AFA179E',
                   '041AFD578A', '041AFD1C27']
    all_colors = ['lightsteelblue', 'mediumseagreen', 'greenyellow', 'salmon',
              'yellow', 'orange']

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
    basal_weights = {'ciri': '151', 'gal': '165', 'joy': '167', 'luna': '152', 'sorrel': '175', 'sparky': '184'}

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

            print(f"Type of row['response_window_end']: {type(row['response_window_end'])}")
            print(f"Type of stim_onset: {type(stim_onset)}")

            response_window_end = row['response_window_end']
            if pd.isna(response_window_end):
                print("response_window_end is NaT; returning NaNs.")
                return np.nan, np.nan, np.nan

            # Convert types as needed
            if isinstance(stim_onset, (float, int)):
                stim_onset = pd.Timestamp(stim_onset, unit='s')
            if isinstance(response_window_end, (float, int)):
                response_window_end = pd.Timestamp(response_window_end, unit='s')

            try:
                max_dur = response_window_end - stim_onset
                if stim_dur_ext <= max_dur:
                    stim_duration = stim_dur_ext
                else:
                    stim_duration = max_dur
                stim_offset = stim_onset + stim_duration
            except TypeError as e:
                print(f"TypeError during max_dur calculation: {e}")
                return np.nan, np.nan, np.nan

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

        # Convert stim_onset and stim_offset to compatible types if necessary
        if isinstance(stim_offset, pd.Timestamp) and isinstance(stim_onset, (float, int)):
            stim_onset = pd.Timestamp(stim_onset, unit='s')
        elif isinstance(stim_onset, pd.Timestamp) and isinstance(stim_offset, (float, int)):
            stim_offset = pd.Timestamp(stim_offset, unit='s')
        elif isinstance(stim_onset, str) or isinstance(stim_offset, str):
            stim_offset = pd.Timestamp(stim_offset, unit='s')
            stim_onset = pd.Timestamp(stim_onset, unit='s')

        # Calculate stim_duration if both values are compatible
        try:
            stim_duration = stim_offset - stim_onset
            return stim_onset, stim_duration, stim_offset
        except TypeError as e:
            print(f"TypeError occurred during subtraction: {e}")
            return np.nan, np.nan, np.nan

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
