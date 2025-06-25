# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')
# model saving
from joblib import dump, load
import joblib

# model loading
rfc9 = joblib.load('/Model/Top9_model.pkl')
# data loading (example-Riaz cohort)
riaz = pd.read_csv("/Data/Riaz_data.csv", sep = "\t", index_col= "Unnamed: 0")
riaz_meta = pd.read_excel("/Data/riaz_metadata.csv", skiprows=2, index_col= "Unnamed: 0")

# visualization
## ROC curve
pred_positive_label = rfc9.predict_proba(riaz)[:, 1]
precisions, recalls, _ = roc_curve(riaz, pred_positive_label)

plt.figure(figsize=(8, 5))
plt.plot([0, 1], [0, 1], label="STR", linestyle="--", color="gray")
plt.plot(precisions_1, recalls_1, label="Gide Cohort", linewidth=2.5, color="#7C9D96")
plt.scatter(precisions_1, recalls_1, color="#7C9D96", s=12)
plt.title("External Validation Cohort ROC Curve(Top8 Model)", fontsize=16)
plt.ylabel("True Positive Rate (TPR)", fontsize=12)
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

## confusion matrix
conf_mat = confusion_matrix(y_true=riaz_metadata,y_pred=pd.DataFrame(rfc9.predict(riaz)))
conf_mat

ax= plt.subplot()
sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax, cmap='PuBu');  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Train Data Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Non-Response', 'Response'])
ax.set_yticklabels(['Non-Response', 'Response'], verticalalignment='center')
