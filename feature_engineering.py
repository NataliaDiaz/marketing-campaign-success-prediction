
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn
from datetime import datetime
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import mstats
from scipy.stats import pearsonr
from scipy.stats import norm
#get_ipython().magic(u'matplotlib inline')

class feature_engineering(object):
	def __init__(self):
		print "feature_engineering methods"

	def plot_histogram(self, data):
	    plt.hist(data)
	    plt.title('Feature histogram')
	    plt.xlabel('Y train histogram')
	    plt.ylabel('Value count')
	    plt.show()

	# testing normality
	def visualize_normality(self, mu, sigma, variable):
	    #mu, sigma = 0, 0.1 # mean and standard deviation
	    s = np.random.normal(mu, sigma, 1000)
	    #Verify the mean and the variance:
	    if not (abs(mu - np.mean(s)) < 0.01):
	        print "normality verification 1 failed"
	    if not abs(sigma - np.std(s, ddof=1)) < 0.01: #ddof=len(variable)-1)) < 0.01:
	        print "normamlity verification 2 failed"
	    #Display the histogram of the samples, along with the probability density function:
	    count, bins, ignored = plt.hist(s, 30, normed=True)
	    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
	        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
	    linewidth=2, color='r')
	    plt.show()


	def visualize_normality_for_sample(self, x):
	    #Display the histogram of the samples, along with the probability density function:
	    mu =  x.mean()
	    sigma = x.std()
	    count, bins, ignored = plt.hist(x, 30, normed=True)
	    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
	        np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
	    linewidth=2, color='r')
	    plt.show()

	def test_normality(self, data):
	    """
	    Tests whether  a sample differs from a normal distribution. Returns a 2-tuple of the chi-squared statistic,
	    and the associated p-value. Given the null hypothesis that x came from a normal distribution,
	    If the p-val is very small (alpha level of 0.05 normally), it means it is unlikely that the data came from a normal distribution.
	    Other possible way: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chisquare.html
	    """
	    # equivalent: print stats.normaltest(data)
	    print "z value and p value: "#, z, pval
	    z,pval = mstats.normaltest(data)
	    if(pval < 0.05):
	        print "Not normal distribution"
	    return z, pval

	# normalization of categorical features
	def create_dummy_vars_for_categorical_features(self, train_df, categorical_features):
	    print "Before dummys:\n", train_df.shape ,' ',train_df.columns.values, '\n', train_df.head()
	    new_df = train_df.copy()
	    for f in categorical_features:
	        num_uniques  = len(train_df[f].unique())
	        df = pd.get_dummies(train_df[f],prefix = f).iloc[:,:num_uniques-1]
	        train_df = pd.concat([train_df, df],axis=1)
	    new_df.drop(categorical_features,axis=1,inplace=True)
	    print "After dummys:\n", df.shape ,' ',train_df.columns.values, '\n',df.head()
	    return new_df


	def find_correlated_features(self, df_full, y, correlation_coef, cols_to_skip=[]):

	    # selecting features to perform pairwise correlation analysis
	    print "initial columns ",df_full.columns.values
	    # selecting features to perform pairwise correlation analysis
	    cols_to_consider = list(df_full.columns.values.copy())
	    for s in cols_to_skip:
	    	cols_to_consider.remove(s)
	    for c in df_full.columns.values :
	    	if c.startswith('1_'):
	    		cols_to_consider.remove(c)
	    print "cols_to_consider", cols_to_consider
	    df =  df_full.loc[:, cols_to_consider]
	    print "Df to find correlation analysis: \n", df.columns.values, '\n', df.head(5)


	    col_names = df.columns.values
	    corr_list = list()
	    for col in range(df.values.shape[1]):
	        x = df.iloc[:,col]
	        _pear =  pearsonr(x, y) # returns the pearson r correlation value, and the p-value associated
	        # also option b) x.corr(y, method='spearman')
	        #print "Column name and Pearson correlation with Y (r and p-value): ",col_names[col], _pear
	        corr_list.append(_pear[0])

	    corr_list = np.array(corr_list)
	    sorting_index = np.argsort(-corr_list)
	    corr_list = corr_list[sorting_index]
	    col_names = col_names[sorting_index]
	    selected_features = list()
	    for (cr, col) in  zip(corr_list,col_names):
	        if abs(cr) > 0.05:
	            selected_features.append((col, cr))
	    print len(selected_features), " features with >0.05 corr coef with Y: \n"
	    for f in selected_features:
	    	print f

	    additional_features = list()
	    mut_correlated_features = []
	    highly_corr_features = []
	    # only if not actually doing feature selection:
	    selected_features = col_names
	    for i in range(len(selected_features)):
	        for j in range(len(selected_features)):
	            if j > i and not (selected_features[i].startswith('1_') and selected_features[j].startswith('1_')):
	                # adding linear combination of features to improve the model
	                feat1_ = selected_features[i]
	                feat2_ = selected_features[j]
	                # comb_feat = feat1_ + '_' + feat2_
	                # additional_features.append(comb_feat)
	                # df[comb_feat] = df[feat1_]*df[feat2_]
	                # detecting features that are correlated with each other
	                # returns the pearson r correlation value, and the p-value associated
	                _pear =  df[feat1_].corr(df[feat2_], method=correlation_coef) #pearsonr(train[feat1_], train[feat2_])
	                if abs(_pear) > 0.01 :
	                    print "Columns HIGHLY correlated (r and p-value) for X1 and X2 ",feat1_, " and ", feat2_, ": ", _pear
	                    highly_corr_features.append((feat1_, feat2_, _pear))
	                else:
	                	print "Columns correlation (r and p-value) for ",feat1_, " and ", feat2_, ": ", _pear
	                # 	mut_correlated_features.append((feat1_, feat2_, _pear[0], _pear[1]))
	  #   for correlation_tuple in highly_corr_features:
			# print "Highly correlated variables corr coeff: ",correlation_tuple
	    print "Highly correlated features among themselves: ",len(highly_corr_features)
	    for f in highly_corr_features:
	    	print f

	def visualize_feature_correlations(self, df_full, correlation_coef, cols_to_skip=[]):
	    """
	    correlation_coef: 'pearson, 'kendall', and 'spearman' are supported. Use pearson when normally distributed data and spearman
	    and kendall when linearity between x and y is observed. Spearman is faster to compute but
	    Kendall is more reliable and interpretable
	    Correlations are returned in a new DataFrame instance (corr_df below).
	    """
	    print "initial columns ",df_full.columns.values
	    # selecting features to perform pairwise correlation analysis
	    cols_to_consider = list(df_full.columns.values.copy())
	    for s in cols_to_skip:
	    	cols_to_consider.remove(s)
	    for c in df_full.columns.values :
	    	if c.startswith('1_'):
	    		cols_to_consider.remove(c)
	    print "cols_to_consider", cols_to_consider
	    df =  df_full.loc[:, cols_to_consider]
	    print "Df to perform correlation analysis: \n", df.columns.values, '\n', df.head(5)

	    # These settings modify the way  pandas prints data stored in a DataFrame.
	    # In particular when we use print(data_frame_reference); function - all
	    #  column values of the frame will be printed in the same  row instead of
	    # being automatically wrapped after 6 columns by default. This will be
	    # for looking at our data at the end of the program.
	    #pd.set_option('display.height', 1000)
	    pd.set_option('display.max_rows', 500)
	    pd.set_option('display.max_columns', 500)
	    pd.set_option('display.width', 1000)

	    corr_df = df.corr(method=correlation_coef)
	    print "--------------- CORRELATIONS ---------------"
	    print corr_df.head(len(df.columns))

	    print "--------------- CREATE A HEATMAP ---------------"
	    # Create a mask to display only the lower triangle of the matrix (since it's mirrored around its
	    # top-left to bottom-right diagonal).
	    mask = np.zeros_like(corr_df)
	    mask[np.triu_indices_from(mask)] = True

	    # Create the heatmap using seaborn library.
	    # List if colormaps (parameter 'cmap') is available here: http://matplotlib.org/examples/color/colormaps_reference.html
	    seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)

	    # Show the plot we reorient the labels for each column and row to make them easier to read.
	    plt.yticks(rotation=0)
	    plt.xticks(rotation=90)
	    plt.show()

	def plot_covariance_based_mutual_info_for_categorical_correlations(self, df, categorical_features = []):
	    """
	    Computes covariance matrix using a vectorized implementation to be used for computing the mutual information
	    coefficient
	    """
	    cols = []
	    plot_id = 1
	    for c in df.columns:
	        if not c.startswith('1_') and c !='0':
	            cols.append(c)
	    plot_grid_wide = len(cols)#/2
	    plot_grid_length = len(cols)#/plot_grid_wide
	    print "Computing covariance matrix and MIC for features: ",cols
	    for i in range(len(cols)):
	        for j in range(len(cols)):
	            if j > i and not (cols[i].startswith('1_') and cols[j].startswith('1_')):
	                #cov_matrix = np.cov([df[cols[i]], df[cols[j]]], ddof= 0)
	                self.MIC_plot(df[cols[i]], df[cols[j]], plot_grid_wide, plot_grid_length, plot_id, cols[i], cols[j])
	                plot_id +=1

	    plt.figure(facecolor='white')
	    #plt.tight_layout()
	    plt.show()

	def MIC_plot(self, x, y, numRows, numCols, plotNum, x_name, y_name, xlim=(-4, 4), ylim=(-4, 4)):
	    # build the MIC and correlation plot using the covariant matrix using a vectorized implementation. To be used when
	    # categorical features are part of the model (otherwise, Pearson, Kendall and Spearman can be used)
	    print "Pearson product-moment correlation coefficients np.corrcoef(x=",x_name,", y=",y_name,"): ",np.corrcoef(x, y)
	    r = np.around(np.corrcoef(x, y)[0, 1], 1)  # Pearson product-moment correlation coefficients.
	    # TODO: compute cov matrix for each one-hot encoding variable of the categorical feature

	    mine = MINE(alpha=0.6, c=15, est="mic_approx")
	    mine.compute_score(x, y)
	    mic = np.around(mine.mic(), 1)
	    ax = plt.subplot(numRows, numCols, plotNum)#,xlim=xlim, ylim=ylim)
	    ax.set_title('Pearson r=%.1f\nMIC=%.1f' % (r, mic),fontsize=10)
	    ax.set_frame_on(False)
	    ax.axes.get_xaxis().set_visible(False)
	    ax.axes.get_yaxis().set_visible(False)
	    ax.plot(x, y, ',')
	    ax.set_xticks([])
	    ax.set_yticks([])
	    return ax
