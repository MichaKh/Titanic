import DataPreProcessor


def main():
    data_train_file_path = 'train.csv'
    data_test_file_path = 'test.csv'
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
    print('Loading train data...')
    train_data_df = DataPreProcessor.load_data(data_train_file_path)
    print('Loading test data...')
    test_data_df = DataPreProcessor.load_data(data_test_file_path)

    print('Cleaning and transforming train data...')
    cleaned_train_data_df = DataPreProcessor.clean_data(train_data_df, data_types)
    print('Cleaning and transforming test data...')
    cleaned_test_data_df = DataPreProcessor.clean_data(test_data_df, data_types)

    print('Saving cleaned train data to file...')
    cleaned_train_data_df.to_csv("clean_train.csv", index=False)
    print('Saving cleaned test data to file...')
    cleaned_test_data_df.to_csv("clean_test.csv", index=False)


if __name__ == '__main__':
    main()
