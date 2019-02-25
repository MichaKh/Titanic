import re

import pandas as pd
import numpy as np


def load_data(data_file_path):
    data_df = pd.read_csv(data_file_path, header=0, index_col=0)
    return data_df


def clean_data(data_df, data_types):
    cleaned_data_df = data_df.copy()
    # Separate last name from first name and affiliation
    cleaned_data_df['Last_Name'] = cleaned_data_df['Name'].apply(lambda name: get_last_name(name), axis=1)
    cleaned_data_df['First_Name'] = cleaned_data_df['Name'].apply(lambda name: get_first_name(name), axis=1)
    cleaned_data_df['Name_Affiliation'] = cleaned_data_df['Name'].apply(lambda name: get_affiliation(name), axis=1)
    cleaned_data_df = cleaned_data_df.drop(columns=['Name'])
    # Correct age
    cleaned_data_df['Age'] = cleaned_data_df['Age'].apply(lambda age: round_age(age), axis=1)
    # Separate ticket number from ticket code
    cleaned_data_df['Ticket_Code'] = cleaned_data_df['Ticket'].apply(lambda ticket: get_ticket_code(ticket), axis=1)
    cleaned_data_df['Ticket_Number'] = cleaned_data_df['Ticket'].apply(lambda ticket: get_ticket_number(ticket), axis=1)
    cleaned_data_df = cleaned_data_df.drop(columns=['Ticket'])
    # Separate cabin floor from number
    cleaned_data_df['Cabin_Floor'] = cleaned_data_df['Cabin'].apply(lambda cabin: get_cabin_floor(cabin), axis=1)
    cleaned_data_df['Cabin_Rooms'] = cleaned_data_df['Cabin'].apply(lambda cabin: get_cabin_number(cabin), axis=1)
    cleaned_data_df = cleaned_data_df.drop(columns=['Cabin'])

    cleaned_data_df = verify_data_types(cleaned_data_df, data_types)
    cleaned_data_df = check_for_missing_values(cleaned_data_df)
    return cleaned_data_df


def verify_data_types(data_df, data_types):
    converters = {
        'Numerical': lambda col: data_df[col].astype(int),
        'Text': lambda col: data_df[col].astype('str'),
        'Categorical': lambda col: data_df[col].astype('category')
    }

    for column in data_df.columns:
        if column in data_types:
            converters[data_types[column]](column)
    return data_df


def check_for_missing_values(data_df):
    for column in data_df:
        if column.dtype in ['int64', 'float64']:
            data_df = data_df[column].loc[data_df[column].isnull()].fillna(np.mean(data_df[column]))
        else:
            data_df = data_df[column].loc[data_df[column].isnull()].fillna('Unknown')
    return data_df


def get_last_name(x):
    if not x or ',' not in x:
        return 'Unknown'
    else:
        name = x.split(',')
        return name[0].strip()


def get_first_name(x):
    if not x or ',' not in x:
        return 'Unknown'
    if '.' not in x:
        return 'Unknown'
    else:
        name = x.split(',')
        first_name_with_affiliation = name[1]
        first_name = first_name_with_affiliation.split('.')
        return first_name[1].strip()


def get_affiliation(x):
    if not x or ',' not in x:
        return 'Unknown'
    if '.' not in x:
        return 'Unknown'
    else:
        name = x.split(',')
        first_name_with_affiliation = name[1]
        first_name = first_name_with_affiliation.split('.')
        return first_name[0].strip().replace('.', '')


def round_age(age):
    return int(np.ceil(age))


def get_ticket_number(x):
    if not x:
        return 0  # Standard ticket
    if ' ' not in x and any(char.isalpha() for char in x):
        return 0
    if ' ' not in x and all(char.isdigit() for char in x):
        return x
    else:
        ticket = x.split(' ')
        return int(ticket[1].strip())


def get_ticket_code(x):
    if not x:
        return 'ST'  # Standard ticket
    if ' ' not in x and all(char.isdigit() for char in x):
        return 'ST'
    else:
        ticket = x.split(' ')
        ticket_code = ''.join(c for c in ticket if c.isalnum())
        return ticket_code


def get_cabin_floor(x):
    all_assigned_cabins = []
    if not x:
        return 'Unknown'
    else:
        all_cabin_nums = x.split(' ')
        for cabin in all_cabin_nums:
            split_cabin_num = re.split('[a-zA-Z]', cabin)
            if len(split_cabin_num) == 1:
                continue
            else:
                all_assigned_cabins.append(split_cabin_num[0].strip())
        return ''.join(set(all_assigned_cabins))


def get_cabin_number(x):
    all_assigned_cabin_numbers = []
    if not x:
        return 'Unknown'
    else:
        all_cabin_nums = x.split(' ')
        for cabin in all_cabin_nums:
            split_cabin_num = re.split('[a-zA-Z]', cabin)
            if len(split_cabin_num) == 1:
                continue
            else:
                all_assigned_cabin_numbers.append(split_cabin_num[1].strip())
        return '-'.join(all_assigned_cabin_numbers)
