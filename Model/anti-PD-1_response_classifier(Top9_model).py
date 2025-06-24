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

# feature importance loading (this was resulted by all model part)
feature_importance = pd.read_csv("/cond1_1_feature_importance.csv")

# parameters -> you can change parameters, but I used this condition.
n_estimators_list = [3]
max_depth_list = [7]
min_samples_leaf_list = [8]
max_features_list = [None]
criterion_list = ['entropy']

parameters = list(itertools.product(n_estimators_list, max_depth_list, min_samples_leaf_list, max_features_list, criterion_list))


# model construction
def RFCmodel_top(X_train, y_train, X_test, y_test, top):

    # feature
    features = feature_importance["Unnamed: 0"].tolist()[:top]
    
    # train data
    X_train = X_train[features]
    
    # test data
    X_test = X_test[features]
    
    train_scores = []
    test_scores = []

    
    for i in range(0, len(parameters)):
    
        # model
        rfc = RandomForestClassifier(n_estimators=parameters[i][0], max_depth=parameters[i][1], min_samples_leaf=parameters[i][2], max_features=parameters[i][3], criterion=parameters[i][4], random_state=0)
        
        # save scores
        # train_score & test_score & 2 valid
        
        # model training
        rfc.fit(X_train, y_train)
        
        # train accuracy, precision, recall, f1 score
        train_acc = round(accuracy_score(y_train, rfc.predict(X_train)), 3)
        train_precision = round(precision_score(y_train, rfc.predict(X_train)), 3)
        train_recall = round(recall_score(y_train, rfc.predict(X_train)), 3)
        train_f1 = round(f1_score(y_train, rfc.predict(X_train)), 3)
        # train scores append
        train_scores.append((train_acc, train_precision, train_recall, train_f1))
        
        # test accuracy, precision, recall, f1 score
        test_acc = round(accuracy_score(y_test, rfc.predict(X_test)), 3)
        test_precision = round(precision_score(y_test, rfc.predict(X_test)), 3)
        test_recall = round(recall_score(y_test, rfc.predict(X_test)), 3)
        test_f1 = round(f1_score(y_test, rfc.predict(X_test)), 3)
        # test scores append
        test_scores.append((test_acc, test_precision, test_recall, test_f1))
        
        
    
    model_result = pd.DataFrame({"parameters":parameters,
                                 "train_scores":train_scores,
                                 "test_scores":test_scores,})    
    return model_result, rfc

model_top_9, rfc9 = RFCmodel_top(X_train, y_train, X_test, y_test, 9)

# model save
joblib.dump(rfc9_check,'/Model/Top9_model.pkl')

# (Gide) validation prediction scores
gide_acc = round(accuracy_score(np.array(gide_meta, dtype=int), rfc.predict(gide)), 3)
gide_precision = round(precision_score(np.array(gide_meta, dtype=int), rfc.predict(gide)), 3)
gide_recall = round(recall_score(np.array(gide_meta, dtype=int), rfc.predict(gide)), 3)
gide_f1 = round(f1_score(np.array(gide_meta, dtype=int), rfc.predict(gide)), 3)
print((gide_acc, gide_precision, gide_recall, gide_f1))

# (Riaz) validation prediction scores
riaz_acc = round(accuracy_score(np.array(riaz_metadata, dtype=int), rfc.predict(riaz)), 3)
riaz_precision = round(precision_score(np.array(riaz_metadata, dtype=int), rfc.predict(riaz)), 3)
riaz_recall = round(recall_score(np.array(riaz_metadata, dtype=int), rfc.predict(riaz)), 3)
riaz_f1 = round(f1_score(np.array(riaz_metadata, dtype=int), rfc.predict(riaz), average="weighted"), 3)
print((riaz_acc, riaz_precision, riaz_recall, riaz_f1))
