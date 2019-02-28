import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_file_path):
    data_df = pd.read_csv(data_file_path, header=0, index_col=0)
    return data_df


def clean_data(data_df, data_types):
    """
    Clean input data.
    Cleaning consists of transforming features (columns) to its right format, filling missing values and verifying the type of each column.
    :param data_df: Data input
    :param data_types: Data types for each column in data
    :return: DataFrame for clean data
    """
    cleaned_data_df = data_df.copy()
    # Separate last name from first name and affiliation
    data_names = cleaned_data_df[['Name', 'Sex']].apply(lambda row: get_affiliation_first_last_name(row), axis=1).apply(pd.Series)
    data_names.columns = ['Name_Affiliation', 'Last_Name', 'First_Name']
    cleaned_data_df = pd.concat([cleaned_data_df[:], data_names.apply(pd.Series)[:]], axis=1)
    cleaned_data_df = cleaned_data_df.drop(labels=['Name'], axis=1)

    # Calculate the number of traveling family members
    cleaned_data_df['Family_Members'] = cleaned_data_df[['SibSp', 'Parch']].apply(lambda sib_parch: get_num_of_family_members(sib_parch), axis=1)

    # Correct age
    cleaned_data_df['Age'] = cleaned_data_df['Age'].apply(lambda age: round_age(age))
    cleaned_data_df['Age_Intervals'] = cleaned_data_df['Age'].apply(lambda age: get_discrete_age_intervals(age))

    # Calculate total fare per passenger
    cleaned_data_df['Fare_Per_Passenger'] = cleaned_data_df[['Fare', 'Family_Members']].apply(lambda fare_family: get_fare_per_passenger(fare_family), axis=1)

    # Separate ticket number from ticket code
    data_tickets = cleaned_data_df['Ticket'].apply(lambda ticket: get_ticket_code_and_number(ticket)).apply(pd.Series)
    data_tickets.columns = ['Ticket_Code', 'Ticket_Number']
    cleaned_data_df = pd.concat([cleaned_data_df[:], data_tickets.apply(pd.Series)[:]], axis=1)
    cleaned_data_df = cleaned_data_df.drop(labels=['Ticket'], axis=1)

    # Separate cabin floor from number
    data_cabins = cleaned_data_df['Cabin'].apply(lambda cabin: get_cabin_floor_and_number(cabin)).apply(pd.Series)
    data_cabins.columns = ['Cabin_Floor', 'Cabin_Rooms']
    cleaned_data_df = pd.concat([cleaned_data_df[:], data_cabins.apply(pd.Series)[:]], axis=1)
    cleaned_data_df = cleaned_data_df.drop(labels=['Cabin'], axis=1)

    cleaned_data_df = check_for_missing_values(cleaned_data_df)
    cleaned_data_df = verify_data_types(cleaned_data_df, data_types)

    assert all(col for col in ~cleaned_data_df.isnull())

    plot_survival_hist(cleaned_data_df['Survived'])
    plot_gender_survival_hist(cleaned_data_df[['Sex', 'Survived']])
    return cleaned_data_df


def plot_survival_hist(survival_data):
    survival_hist = survival_data.value_counts().sort_index().plot(kind="bar", alpha=0.5)
    plt.title("Titanic Survival Histogram")
    plt.xlabel("Survived?")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def plot_gender_survival_hist(survival_data):
    ax = survival_data.groupby(['Sex', 'Survived']).size().unstack().plot(kind='bar', stacked=True)
    plt.title("Titanic Gender Survival Histogram")
    plt.xlabel("Sex")
    plt.ylabel("Frequency")
    ax.legend(["Not Survived", "Survived"])
    plt.show()


def verify_data_types(data_df, data_types):
    """
    Convert all column in dataframe to its correct types (Object, int32, or category)
    :param data_df: Data input
    :param data_types: Data types for each column in data
    :return: DataFrame for clean data
    """
    converters = {
        'Numerical': lambda df, col: df[col].astype(int),
        'Text': lambda df, col: df[col].astype('str'),
        'Categorical': lambda df, col: df[col].astype('category')
    }

    for column in data_df.columns:
        if column in data_types:
            data_df[column] = converters[data_types[column]](data_df, column)
    return data_df


def check_for_missing_values(data_df):
    """
    Fill missing values according to the type of each column
    :param data_df: Data input
    :return: DataFrame for clean data
    """
    for column in data_df:
        missing_data = data_df[column].loc[data_df[column].isnull()]
        if len(missing_data) > 0:
            if data_df[column].dtype in ['int64', 'float64']:
                data_df[column] = data_df[column].fillna(np.mean(data_df[column]))
            else:
                data_df[column] = data_df[column].fillna('Unknown')
    return data_df


def get_affiliation_first_last_name(x):
    """
    Separate written name from first, last and personal affiliation (e.g., Braund, Mr. Owen Herris ->  Mr. Braund, Owen Herris)
    :param x: Name
    :return: [<Personal affiliation>, <Last Name>, <First Name>]
    """
    if (not x['Name']) or (',' not in x['Name']):
        return ['Unknown', 'Unknown', 'Unknown']
    else:
        name = x['Name'].split(',')
        sex = x['Sex'] if x['Sex'] else 'female'
        last_name = name[0].strip()
        first_name_with_affiliation = name[1]
        first_name_with_affiliation_list = first_name_with_affiliation.split('.')
        first_name = first_name_with_affiliation_list[1].split('(')[0].strip()
        if not first_name:
            first_name = 'Unknown'
        if '.' not in x['Name']:
            affiliation = 'Unknown'
        else:
            affiliation = first_name_with_affiliation_list[0].strip().replace('.', '')
            if affiliation in ['Capt', 'Don', 'Major', 'Sir', 'Jonkheer', 'Rev', 'Col', 'Master']:
                affiliation = 'Master'
            elif affiliation in ['the Countess', 'Mlle', 'Mme']:
                affiliation = 'Mrs'
            elif affiliation in ['Mlle', 'Ms', 'Lady', 'Miss']:
                affiliation = 'Miss'
            elif affiliation in ['Dr']:
                affiliation = 'Mr' if sex == 'male' else 'Mrs'
        return [affiliation, last_name, first_name]


def round_age(age):
    try:
        return int(np.ceil(age))
    except ValueError:
        return


def get_discrete_age_intervals(age):
    try:
        rounded_age = int(np.ceil(age))
        if rounded_age <= 20:
            return 'Young'
        elif 20 < rounded_age <= 65:
            return 'Adult'
        else:
            return 'Elderly'
    except ValueError:
        return 'Adult'


def get_num_of_family_members(x):
    sib_sp = x['SibSp']  # num of siblings or spouses
    parch = x['Parch']  # num of parents or children
    sib_sp = sib_sp if not np.isnan(sib_sp) else 0
    parch = parch if not np.isnan(parch) else 0

    return sib_sp + parch + 1


def get_fare_per_passenger(x):
    total_fare = x['Fare']
    num_of_family_members = x['Family_Members']
    total_fare = total_fare if not np.isnan(total_fare) else 0
    num_of_family_members = num_of_family_members if not np.isnan(num_of_family_members) else 1
    return int(total_fare / (num_of_family_members + 1))


def get_ticket_code_and_number(x):
    """
    Separate ticket code from ticket number (e.g., PC 144254 -> code PC number 144254)
    :param x: Ticket
    :return: [<Ticket code>, <Ticket number>]
    """
    if not x:
        return 'ST', 0
    if ' ' not in x and all(char.isdigit() for char in x):
        return 'ST', x
    if ' ' not in x and any(char.isalpha() for char in x):
        return x, 0
    ticket = x.split(' ')
    if len(ticket) == 3:
        ticket[0] = ticket[0] + ticket[1]
        ticket[1] = ticket[-1]
    ticket_code = ''.join(c for c in ticket[0] if c.isalnum()).upper()
    ticket_number = int(ticket[1].strip())
    return [ticket_code, ticket_number]


def get_cabin_floor_and_number(x):
    """
    Separate cabin floor from cabin number (e.g., C45 -> floor C room 45)
    :param x: Cabin
    :return: [<Cabin floor>, <Cabin number>]
    """
    all_assigned_cabins = []
    all_assigned_cabin_numbers = []
    if pd.isnull(x):
        return 'Unknown', 'Unknown'
    else:
        all_cabin_nums = x.split(' ')
        for cabin in all_cabin_nums:
            split_cabin_num = re.findall(r'[^\W\d_]+|\d+', cabin)
            deck = split_cabin_num[0].strip()
            if deck in ['S', 'A', 'B', 'C', 'D']:
                all_assigned_cabins.append('Upper')
            elif deck in ['E']:
                all_assigned_cabins.append('Main')
            elif deck in ['F']:
                all_assigned_cabins.append('Middle')
            else:
                all_assigned_cabins.append('Lower')
            if len(split_cabin_num) == 1:
                all_assigned_cabin_numbers.append('0')
            else:
                all_assigned_cabin_numbers.append(split_cabin_num[1].strip())
        return [''.join(set(all_assigned_cabins)), '-'.join(all_assigned_cabin_numbers)]
