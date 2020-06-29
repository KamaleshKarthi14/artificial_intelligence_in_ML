if __name__ == "__main__":

    #Importing some libraries
    import numpy as np
    import pandas as pd
    import os
    #Getting rid of pesky warnings
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    np.warnings.filterwarnings('ignore')

    column_names = [
         	"Age",
		"BusinessTravel",	
		"Department",
		"DistanceFromHome",
		"Education",
		"EnvironmentSatisfaction",
		"Gender",
		"JobInvolvement",
		"obLevel",
		"JobRole",
		"JobSatisfaction",
		"MaritalStatus",
		"MonthlyIncome",
		"NumCompaniesWorked",
		"OverTime",
		"PercentSalaryHike",
		"PerformanceRating",
		"StockOptionLevel",
		"TotalWorkingYears",
		"TrainingTimesLastYear",
		"WorkLifeBalance",
		"YearsAtCompany",
		"YearsInCurrentRole",
		"YearsSinceLastPromotion",
		"YearsWithCurrManager"
                ]
    #Importing the dataset
    location = 'final.csv'
    dataset = pd.read_csv(location)
    #X = dataset.iloc[:, [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]].values
    #Y = dataset.iloc[:, 2].values
    dataset = dataset.drop(['Unnamed: 0'],axis=1)
    X=dataset.iloc[:,dataset.columns !='Attrition']
    print(X.columns)
    Y=dataset.iloc[:,dataset.columns =='Attrition']
    #Replace all 'heart-disease' values greater than 0 because my goal is not to classify the disease type
    #for x,i in enumerate(Y):
     #   if i>0:Y[x]=1
    #Splitting the dataset into the Training set and Test set
    from sklearn.model_selection._split import train_test_split
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN()
    X_resampled, y_resampled = smote_enn.fit_sample(X, Y)
    X_train, X_test, Y_Train, Y_Test = train_test_split(X_resampled, y_resampled, test_size=0.25)

    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    #Using Pipeline
    import sklearn.pipeline
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import KernelPCA
    from imblearn.pipeline import make_pipeline
    
    #select = sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)
    clf = MLPClassifier(solver='lbfgs', learning_rate='constant', activation='tanh')
    kernel = KernelPCA()
    
    pipeline = make_pipeline(kernel, clf)
    pipeline.fit(X_train, Y_Train)

    #User-input
    v = []
    for i in column_names[:]:
        v.append(input(i+": "))
    answer = np.array(v)
    answer = answer.reshape(1,-1)
    answer = sc_X.transform(answer)
    print ("Predicts:"+ str(pipeline.predict(answer)))
    #print ("("Predicts: " + str(pipeline.predict(answer))")
