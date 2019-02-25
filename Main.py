import DataPreProcessor


def main():
    data_file_path = 'train.csv'
    data_types = {'Survived': 'Categorical',
                  'Pclass': 'Categorical',
                  'Name_Affiliation': 'Categorical',
                  'Last_Name': 'Text',
                  'First_Name': 'Text',
                  'Sex': 'Categorical',
                  'Age': 'Numerical',
                  'SibSp': 'Numerical',
                  'Parch': 'Numerical',
                  'Ticket_Code': 'Categorical',
                  'Ticket_Number': 'Numerical',
                  'Fare': 'Numerical',
                  'Cabin_Floor': 'Categorical',
                  'Cabin_Rooms': 'Text',
                  'Embarked': 'Categorical'}
    DataPreProcessor.load_data(data_file_path, data_types)


if __name__ == '__main__':
    main()
