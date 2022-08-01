# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 16:37:20 2021

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% load prepared data
df = pd.read_pickle('prepared_dataframe.pickle')

# %% get a female and a male actor and compare

act01 = df[ (df['actor_ID'] == '01') & (df['phrase_ID'] == '02') ]
act02 = df[ (df['actor_ID'] == '02') & (df['phrase_ID'] == '02') ]
folder_for_figs = 'act_01_02'

# %% or another pair

act01 = df[ (df['actor_ID'] == '03') & (df['phrase_ID'] == '02') ]
act02 = df[ (df['actor_ID'] == '04') & (df['phrase_ID'] == '02') ]
folder_for_figs = 'act_03_04'

# %% male - female

 #act01 = df[ (df['female'] == True) & (df['phrase_ID'] == '02') ]
 #act02 = df[ (df['female'] == False) & (df['phrase_ID'] == '02') ]
#act01 = df[ df['female'] == True ]
#act02 = df[ df['female'] == False ]
#folder_for_figs = 'fe-male'

# %% here
act01 = df[ df['emotion'] == 'angry' ]
act02 = df[ df['emotion'] == 'calm' ]
folder_for_figs = 'calm_angry'
#%%Isolate features and labels for 24
act01_feat = np.vstack( act01['mfcc_profile'].to_numpy() )
act02_feat = np.vstack( act02['mfcc_profile'].to_numpy() )


act01_featb= np.vstack(act01[['mean_centroid', 'std_centroid', 'mean_bandwidth', 'std_bandwidth']].to_numpy())
act02_featb = np.vstack(act02[['mean_centroid', 'std_centroid', 'mean_bandwidth', 'std_bandwidth']].to_numpy())

act01_features = np.c_[ act01_featb, act01_feat ]
act02_features = np.c_[ act02_featb , act02_feat ]

all_features = np.vstack((act01_features, act02_features))


act01_labels = 0*np.ones( ( act01_features.shape[0] , 1 ) )
act02_labels = 1*np.ones( ( act02_features.shape[0] , 1 ) )
all_labels = np.r_[ act01_labels , act02_labels ]

# %% isolate features and labels for 20

#act01_features = np.vstack( act01['mfcc_profile'].to_numpy() )
#act02_features = np.vstack( act02['mfcc_profile'].to_numpy() )
#all_features = np.vstack((act01_features, act02_features))
#
#act01_labels = 0*np.ones( ( act01_features.shape[0] , 1 ) )
#act02_labels = 1*np.ones( ( act02_features.shape[0] , 1 ) )
#all_labels = np.r_[ act01_labels , act02_labels ]

# %% apply and plot with PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_features = np.vstack( all_features )
all_pca = pca.fit_transform( np.vstack( pca_features ) )

plt.clf()
plt.plot( all_pca[:act01_features.shape[0], 0] , all_pca[:act01_features.shape[0], 1] , 'bx', alpha=0.8 )
plt.plot( all_pca[act01_features.shape[0]:, 0] , all_pca[act01_features.shape[0]:, 1] , 'r+', alpha=0.8 )
plt.savefig('figs/' + folder_for_figs + '/pca.png', dpi=300)

# %% apply and plot with MDS

from sklearn.manifold import MDS

mds = MDS(n_components=2)
mds_features = np.vstack( all_features )
all_mds = mds.fit_transform( np.vstack( mds_features ) )

plt.clf()
plt.plot( all_mds[:act01_features.shape[0], 0] , all_mds[:act01_features.shape[0], 1] , 'bx', alpha=0.8 )
plt.plot( all_mds[act01_features.shape[0]:, 0] , all_mds[act01_features.shape[0]:, 1] , 'r+', alpha=0.8 )
plt.savefig('figs/' + folder_for_figs + '/mds.png', dpi=300)

# %% apply and plot with TSNE

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne_features = np.vstack( all_features )
all_tsne = tsne.fit_transform( np.vstack( tsne_features ) )

plt.clf()
plt.plot( all_tsne[:act01_features.shape[0], 0] , all_tsne[:act01_features.shape[0], 1] , 'bx', alpha=0.8 )
plt.plot( all_tsne[act01_features.shape[0]:, 0] , all_tsne[act01_features.shape[0]:, 1] , 'r+', alpha=0.8 )
plt.savefig('figs/' + folder_for_figs + '/tsne.png', dpi=300)

# %% train - test split

from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split( np.c_[ all_features , all_labels] , test_size=0.2 , random_state=42 )

train_input = train_set[:, :-1]
train_label = train_set[:, -1]
test_input = test_set[:, :-1]
test_label = test_set[:, -1]

# %% linear regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit( train_input , train_label )
# make predictions from training data
preds = lin_reg.predict( test_input )
preds_binary = np.array( preds >= 0.5 ).astype(int)
comparison_check = np.c_[ preds , preds_binary , test_label ]
accuracy_linear = np.sum( test_label == preds_binary ) / preds.size

# %% random forest

from sklearn.ensemble import RandomForestClassifier

forest_class = RandomForestClassifier()
forest_class.fit( train_input , train_label )
# make predictions from training data
preds_binary = forest_class.predict( test_input )
comparison_check = np.c_[ preds_binary , test_label ]
accuracy_forest = np.sum( test_label == preds_binary ) / preds.size

# %% SVM

from sklearn.svm import SVC

svm_class = SVC()
svm_class.fit( train_input , train_label )
# make predictions from training data
preds_binary = svm_class.predict( test_input )
comparison_check = np.c_[ preds_binary , test_label ]
accuracy_svm = np.sum( test_label == preds_binary ) / preds.size

# %% cross validation - custom accuracy metric

from sklearn.metrics import make_scorer

def binary_accuracy( y_true , y_pred ):
    bin_pred = np.array( y_pred >= 0.5 ).astype(int)
    return np.sum( y_true == bin_pred ) / y_true.size

my_scorer = make_scorer(binary_accuracy, greater_is_better=True)

# %% cross validation

from sklearn.model_selection import cross_val_score

scores_lin = cross_val_score( lin_reg, all_features, all_labels,
                         scoring=my_scorer, cv=10 )

scores_forest = cross_val_score( forest_class, all_features, all_labels.ravel(),
                         scoring=my_scorer, cv=10 )

scores_svm = cross_val_score( svm_class, all_features, all_labels.ravel(),
                         scoring=my_scorer, cv=10 )

def present_scores( s , algorithm='method' ):
    print(30*'-')
    print( algorithm + ' accuracy in 10-fold cross validation:' )
    print('mean: ' + str( np.mean(s) ))
    print('std: ' + str( np.std(s) ))
    print('median: ' + str( np.median(s) ))

present_scores( scores_lin , algorithm='linear regression' )
present_scores( scores_forest , algorithm='random forest' )
present_scores( scores_svm , algorithm='SVM' )