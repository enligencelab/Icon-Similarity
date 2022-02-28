import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

applications = {
    'CNN-SSIM': {
        'valid_path': "raw/app1_validation.xlsx",
    },
}

fpr_poly_ = np.linspace(0, 1, 200)
for app_key in applications.keys():
    validation = pd.read_excel(applications[app_key]['valid_path'])
    y_pred = validation['Estimated'].values
    y_true = validation['Ground Truth'].values
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    tpr_poly_ = np.interp(fpr_poly_, fpr, tpr)
    plt.plot(fpr_poly_, tpr_poly_, label=f'{app_key} (AUC = {auc.__round__(3)})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=1.25, color='b', label='Chance')
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.legend(loc=4)
plt.savefig('results/compare_roc.eps')
plt.close()
