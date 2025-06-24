# model generation & test
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# model saving
from joblib import dump, load
import joblib

# data loading
liu = pd.read_csv("/Data/Liu_data.csv")
gide = pd.read_csv("/Data/Gide_data.csv", sep = "\t", index_col= "Unnamed: 0")
riaz = pd.read_csv("/Data/Riaz_data.csv", sep = "\t", index_col= "Unnamed: 0")
# metadata loading
liu_meta = pd.read_excel("/Data/Liu_metadata.csv", skiprows=2, index_col= "Unnamed: 0")
gide_meta = pd.read_excel("/Data/gide_metadata.csv", skiprows=2, index_col= "Unnamed: 0")
riaz_meta = pd.read_excel("/Data/riaz_metadata.csv", skiprows=2, index_col= "Unnamed: 0")

# saving list
mean_train = []
mean_test = []
save_model = []

# parameters -> you can change parameters, but I used this condition.
n_estimators_list = [5]
max_depth_list = [5]
min_samples_leaf_list = [12]
max_features_list = ['log2'] 
criterion_list = ['gini']
parameters = list(itertools.product(n_estimators_list, max_depth_list, min_samples_leaf_list, max_features_list, criterion_list))


# model    
for i in range(0, len(parameters)):

    # cv = 5, train 0.8, test 0.2
    stratified_shuffle_split = StratifiedShuffleSplit(n_splits=5, train_size=0.8, test_size=0.2, random_state=2024)

    # rfc
    rfc = RandomForestClassifier(n_estimators=parameters[i][0], max_depth=parameters[i][1], min_samples_leaf=parameters[i][2], max_features=parameters[i][3], criterion=parameters[i][4], random_state=0)

    # train_score & test_score
    train_scores = []
    test_scores = []

    # StratifiedShuffleSplit train data & test data
    for train_index, test_index in stratified_shuffle_split.split(pd.DataFrame(X), y):

        # train, test data split
        X_train, X_test = pd.DataFrame(X).iloc[train_index], pd.DataFrame(X).iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # model training
        rfc.fit(X_train, y_train)
        save_model.append(rfc)

        # train accuracy_score
        train_score = accuracy_score(y_train, rfc.predict(X_train))
        train_scores.append(train_score)

        # test accuracy_score
        test_score = accuracy_score(y_test, rfc.predict(X_test))
        test_scores.append(test_score)
    
    # mean_train, mean_test saving
    mean_train.append(np.mean(train_scores))
    mean_test.append(np.mean(test_scores))
    
    model_result = pd.DataFrame({"parameters":parameters,
                                "train_scores":mean_train,
                                "test_scores":mean_test})
    
## after training, we validated exterenal cohorts.
# validation prediction scores
valid_scores = accuracy_score(np.array(gide_metadata,dtype=int), rfc.predict(gide))
print("Gide_acc score :" valid_scores)

# validation prediction scores
valid_scores_2 = accuracy_score(np.array(Riaz_metadata,dtype=int), rfc.predict(Riaz))
print("Riaz_acc score :" valid_scores_2)
