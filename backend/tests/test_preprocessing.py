# from data_cleansing import cleaning_data
# from data_spliting import spliting_data

# ## load csv 
# FILE_NAME = "../database/Salary_Data.csv"
# df = pd.read_csv(FILE_NAME, delimiter=',')
# df = cleaning_data(df, has_target_columns=True)
# X_train, X_test, y_train, y_test = spliting_data(df)
# print(X_train)
# print(X_test)

# # # test 1
# # X_train_, X_test_ = preprocess_data(X_train, y_train, X_test,
# #                                     use_polynomial=True)
# # print(X_train_)
# # print(X_test_)
# # test 2
# X_train_, X_test_ = preprocess_data(X_train, y_train, X_test, use_polynomial=False)
# print(X_train_)
# print(X_test_)

# exam = pd.DataFrame([{
#     'age': 20,
#     'gender': 'female',
#     'education_level': 'PhD',
#     'job_title': 'Data Engineer',
#     'years_of_experience': 1,
# }])
# # test 3
# _, exam_ = preprocess_data(X_train, y_train, exam, use_polynomial=True)
# print(exam_)
# # test 4
# _, exam_ = preprocess_data(X_train, y_train, exam, use_polynomial=False)
# print(exam_)
pass