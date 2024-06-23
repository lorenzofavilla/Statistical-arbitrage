import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

#prices
P=pd.read_excel("Eurostoxx_financials_returns.xlsx",sheet_name=1,index_col=0,parse_dates=True)
P.index=P.index.strftime('%Y-%m-%d')

#weekly prices and train test split
wP=P.iloc[::5]
Ptrain=wP.values[:256,:]
Ptest=wP.values[256:,:]

#weekly linear returns and train test split
wR= wP.pct_change().dropna()
Rtrain=wR.values[:256,:]
Rtest=wR.values[256:,:]

#corr matrix
Rcorr=np.corrcoef(Rtrain,rowvar=False)
Rcorr_df=pd.DataFrame(Rcorr,index=P.columns,columns=P.columns)

#fill main diagonal (1s) with NaN
np.fill_diagonal(Rcorr_df.values, np.nan)
print(Rcorr_df)

#get max corr and loc
max_corr=Rcorr_df.max().max()
max_corr_row,max_corr_col=np.where(Rcorr_df==max_corr)

#compute statistics
mean_corr=np.nanmean(Rcorr_df)
sd_corr=np.nanstd(Rcorr_df)
min_corr=np.nanmin(Rcorr_df)
max_corr=np.nanmax(Rcorr_df)

#print results
print("Avg corr: ",mean_corr)
print("SD: ", sd_corr)
print("Min corr: ",min_corr)
print("Max corr: ",max_corr)

print("Max row label: ",Rcorr_df.index[max_corr_row])
print("Max column label: ",Rcorr_df.index[max_corr_col])

#extract price data of max correlated series
maxCorrTrainP=Ptrain[:,[max_corr_row[0],max_corr_col[0]]]
maxCorrTestP=Ptest[:,[max_corr_row[0],max_corr_col[0]]]

#extract return data of max correlated series
maxCorrTrainR=Rtrain[:,[max_corr_row[0],max_corr_col[0]]]
maxCorrTestR=Rtest[:,[max_corr_row[0],max_corr_col[0]]]





#OLS REGRESSION ON PRICES
#intercept not forced to the origin
reg=LR(fit_intercept=True)

#regress second column on first column (weekly prices of 2 most correlated series)
regfit=reg.fit(maxCorrTrainP[:,1].reshape(-1, 1), maxCorrTrainP[:,0])

#values of independent variable
YtrainP=maxCorrTrainP[:,0]
YtestP=maxCorrTestP[:,0]

#residuals (actual - fitted) of independent variable
eYtrainP=YtrainP-reg.predict(maxCorrTrainP[:,1].reshape(-1, 1))
eYtestP=YtestP-reg.predict(maxCorrTestP[:,1].reshape(-1, 1))

#R-squared
print("R-squared for prices:")
print("Train: ",reg.score(maxCorrTrainP[:,1].reshape(-1, 1), maxCorrTrainP[:,0])*100,"%")
print("Test: ",reg.score(maxCorrTestP[:,1].reshape(-1, 1), maxCorrTestP[:,0])*100,"%")

#coefficient
beta=regfit.coef_

#first difference of residuals
etrainDiffP=np.diff(eYtrainP,n=1,axis=0)
etestDiffP=np.diff(eYtestP,n=1,axis=0)

#PLOTS
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))

#plot of residuals of train
axs[0].plot(eYtrainP)
axs[0].set_title("Price residuals for train set")

#plot of residuals of test
axs[1].plot(eYtestP)
axs[1].set_title("Price residuals for test set")

#acf plot of first difference of trainining set residuals
plot_acf(etrainDiffP, ax=axs[2])
axs[2].set_title("ACF of first diff train error")

#acf plot of first difference of testing set residuals
plot_acf(etestDiffP, ax=axs[3])
axs[3].set_title("ACF of first diff test error")

plt.tight_layout()
plt.show()





#OLS REGRESSION ON LINEAR RETURNS
#intercept not forced to the origin
reg=LR(fit_intercept=True)

#regress second column on first column (weekly linear returns of 2 most correlated series)
regfit=reg.fit(maxCorrTrainR[:,1].reshape(-1, 1), maxCorrTrainR[:,0])

#values of independent variable
YtrainR=maxCorrTrainR[:,0]
YtestR=maxCorrTestR[:,0]

#residuals (actual - fitted) of independent variable
eYtrainR=YtrainR-reg.predict(maxCorrTrainR[:,1].reshape(-1, 1))
eYtestR=YtestR-reg.predict(maxCorrTestR[:,1].reshape(-1, 1))

#R-squared
print("R-squared for linear returns:")
print("Train: ",reg.score(maxCorrTrainR[:,1].reshape(-1, 1), maxCorrTrainR[:,0])*100,"%")
print("Test: ",reg.score(maxCorrTestR[:,1].reshape(-1, 1), maxCorrTestR[:,0])*100,"%")

#coefficient
beta=regfit.coef_

#initialize price index matrices for both fit and test datasets
price_indexTrain=np.ones((eYtrainR.shape[0] + 1, eYtrainR.shape[0]))
price_indexTest=np.ones((eYtestR.shape[0] + 1, eYtestR.shape[0]))

#calculate price indices by cumulatively applying percent changes derived from residuals
for i in range(1,eYtrainR.shape[0]+1):
    price_indexTrain[i]=price_indexTrain[i-1]*(1+eYtrainR[i-1])

for i in range(1,eYtestR.shape[0]+1):
    price_indexTest[i]=price_indexTest[i-1]*(1+eYtestR[i-1])

#first difference of residuals
etrainDiffR=np.diff(eYtrainR,n=1,axis=0)
etestDiffR=np.diff(eYtestR,n=1,axis=0)

#PLOTS
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))

#plot of price index based on train residuals
axs[0].plot(price_indexTrain)
axs[0].set_title("Training price index")

#plot of price index based on test residuals
axs[1].plot(price_indexTest)
axs[1].set_title("Testing price index")

#acf plot of first difference of trainining set residuals
plot_acf(etrainDiffR, ax=axs[2])
axs[2].set_title("ACF of first diff train error")

#acf plot of first difference of testing set residuals
plot_acf(etestDiffR, ax=axs[3])
axs[3].set_title("ACF of first diff test error")

plt.tight_layout()
plt.show()
