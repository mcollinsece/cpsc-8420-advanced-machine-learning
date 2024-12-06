import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colors

df = pd.read_csv("default_credit_card_clients.csv", header=1)
df.head()
df.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
df.head()
df.drop(columns=['ID'], inplace=True)
df.head()
df.dtypes
df['SEX'].unique()
df['EDUCATION'].unique()
df['MARRIAGE'].unique()
len(df.loc[df['EDUCATION'] == 0]) | len(df.loc[df['MARRIAGE'] == 0])
len(df)
df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
len(df_no_missing)
df_no_missing['EDUCATION'].unique()
df_no_missing['MARRIAGE'].unique()
df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]
df_no_default_downsampled = resample(df_no_default, replace=False, n_samples=1000, random_state=42)
len(df_no_default_downsampled)
df_default_downsampled = resample(df_default, replace=False, n_samples=1000, random_state=42)
len(df_default_downsampled)
df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled])
len(df_downsample)
X = df_downsample.drop("DEFAULT", axis=1).copy()
X.head()
y = df_downsample['DEFAULT'].copy()
y.head()
pd.get_dummies(X,columns=['MARRIAGE'], dtype=int).head()
X_encoded = pd.get_dummies(X,columns=['SEX', 
                                      'EDUCATION', 
                                      'MARRIAGE', 
                                      'PAY_0',
                                      'PAY_2',
                                      'PAY_3',
                                      'PAY_4',
                                      'PAY_5',
                                      'PAY_6'
                                      ], dtype=int)
X_encoded.head()
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train)
ConfusionMatrixDisplay.from_estimator(
    clf_svm,
    X_test_scaled,
    y_test,
    values_format='d',
    display_labels=['Did not default', 'Defaulted']
)
param_grid = {
    'C': [0.5, 1, 10, 100],
    'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}  
optimal_params = GridSearchCV(SVC(),
                              param_grid,
                              cv=5,
                              scoring='accuracy',
                              verbose=0)
optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)
clf_svm = SVC(random_state=42, C=1, gamma=0.01)
clf_svm.fit(X_train_scaled, y_train)
ConfusionMatrixDisplay.from_estimator(
    clf_svm,
    X_test_scaled,
    y_test,
    values_format='d',
    display_labels=['Did not default', 'Defaulted']
)
len(df_downsample.columns)
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    labelbottom=False
)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.show()
train_pc1_coords = X_train_pca[:, 0]
train_pc2_coords = X_train_pca[:, 1]

pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

param_grid = [
    {'C': [1, 10, 100, 1000],
     'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
     'kernel': ['rbf']

    }
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)
clf_svm = SVC(random_state=42, C=1000, gamma=0.001)
clf_svm.fit(pca_train_scaled, y_train)

X_test_pca = pca.transform(X_train_scaled)
test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]

x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1

y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1))

Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10,10))
ax.contourf(xx, yy, Z, alpha=0.1)

cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])

scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train,
                     cmap=cmap,
                     s=100,
                     edgecolors='k',
                     alpha=0.7)

legend = ax.legend(scatter.legend_elements()[0],
                   scatter.legend_elements()[1],
                   loc="upper right")
legend.get_texts()[0].set_text("No Default")
legend.get_texts()[1].set_text("Yes Default")

ax.set_ylabel("PC2")
ax.set_xlabel("PC1")
ax.set_title('Decision surface using the PCA transformed features')
plt.show()



