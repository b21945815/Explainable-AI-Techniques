from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 61)

explainer = LimeTabularExplainer(
    training_data=X_train, 
    mode='classification', 
    feature_names=iris.feature_names, 
    class_names=iris.target_names, 
    discretize_continuous=True
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

sample_index = 0 
sample = X_test[sample_index]

explanation = explainer.explain_instance(sample, model.predict_proba, num_features=4)

exp_data = explanation.as_list()
exp_df = pd.DataFrame(exp_data, columns=["Feature", "Contribution"])
print("LIME Explanation for Sample Index", sample_index)
print(exp_df)

exp_df.plot.barh(x="Feature", y="Contribution", legend=False, color="skyblue")
plt.xlabel("Contribution")
plt.ylabel("Feature")
plt.title("LIME Explanation")
plt.show()

# SHAP ile açıklama
explainer = shap.KernelExplainer(model.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
force_plot = shap.force_plot(explainer.expected_value[0], shap_values[..., 0], X_test)
shap.save_html("shap_force_plot.html", force_plot)