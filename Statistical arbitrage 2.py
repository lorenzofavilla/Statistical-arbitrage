import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler
import random

#prices
P=pd.read_excel("Eurostoxx_financials_returns.xlsx",header=0,sheet_name=1,index_col=0,parse_dates=True)
P.index=P.index.strftime('%Y-%m-%d')

#weekly prices and train test split
wP=P.iloc[::5]
Ptrain=wP.values[:256,:]
Ptest=wP.values[256:,:]
row_index=P.columns

#weekly linear returns and train test split
wR= wP.pct_change().dropna()
Rtrain=wR.values[:256,:]
Rtest=wR.values[256:,:]





#ROLLING REGRESSION FUNCTION
def rolling_regression_sklearn(data,q,f):
    n_rows=data.shape[0]
    e_forecasts=np.zeros(n_rows)
    no_trade=np.zeros(n_rows)

    for i in range(int((n_rows - q)/f)):            #in 1st plot: f=10 and q=52 --> range from 1 to (512-52)/10=45 
        start_idx = i*f                             #1st loop: from 10 to 62
        end_idx = i*f + q                           #2nd loop: from 20 to 72... up to 45th loop: from 450 to 502
        
        X = data[start_idx:end_idx, 1:]             #independent variables X (from column 1 onwards)
        y = data[start_idx:end_idx, 0]              #dependent variable y (only column 0)
                                                    #rows from start_idx to end_idx (regressing 1st column (BNP) on all other on rolling basis)
        #intercept forced to the origin
        reg = LR(fit_intercept=False)
        reg.fit(X,y)
        #print("R-square (fit) from week ",start_idx,"to week ",end_idx,": ",reg.score(X,y))

        #forecasting
        forecast_X = data[end_idx:end_idx+f, 1:]    #future regressors
        forecast = reg.predict(forecast_X)          #forecast dependent variable (fitted value)
        no_trade[end_idx]=1                         #no_trade on first day of forecasting

        #calculating forecast errors
        forecast_y = data[end_idx:end_idx+f, 0]     #future dep variable (actual value)
        forecast_errors = (forecast_y - forecast)   #difference between fitted and actual values
        e_forecasts[end_idx:end_idx+f]=(forecast_errors)

    return e_forecasts,no_trade

#1st plot: residuals
data=wP.values              
data=np.delete(data,[14],axis=1)                    #drop Montepaschi
q=52                                                #train weeks
f=10                                                #test weeks
n_rows=data.shape[0] 
errors, no_trade = rolling_regression_sklearn(data[:,0:], q, f)

plt.plot(errors)     
plt.title('Forecast errors (PLOT 1)')
plt.xlabel('Weeks')
plt.ylabel('Forecast errors')
plt.show()

#2nd plot: mean reverting strategy
pl=np.zeros(n_rows)
for i in range (q,n_rows):
    if no_trade[i]==1:
        pl[i]=0
    if no_trade[i]==0:
        pl[i]=-np.sign(errors[i-1])*(errors[i]-errors[i-1])

plt.plot(pl)
plt.title('Trading Strategy P&L (PLOT 2)')
plt.xlabel('Weeks')
plt.ylabel('Weekly P&L')
plt.show()

#3rd plot: cumulative P&L
sumpl = np.cumsum(pl)
plt.plot(sumpl)
plt.title('Cumulative P&L (PLOT 3)')
plt.xlabel('Weeks')
plt.ylabel('Cumulative P&L')
plt.show()





#PRINCIPAL COMPONENTS FUNCTION
def rolling_PCA_sklearn(data,q,f,k1):
    n_rows=data.shape[0]
    e_forecasts=np.zeros(n_rows)
    no_trade=np.zeros(n_rows)

    for i in range(int((n_rows - q)/f)):
        start_idx = i*f                             
        end_idx = i*f + q

        #extract the data window for PCA
        X = data[start_idx:end_idx,:]

        #perform PCA on the data subset
        pca = PCA(n_components=data.shape[1])
        fit_X = pca.fit_transform(X)          

        #sum the components with the least variance determined by 'k1', calculate mean
        fit_X = np.sum(fit_X[:, -k1:], axis=1)
        mean_X=np.mean(fit_X)

        #forecast using PCA
        forecast_X = data[end_idx:end_idx + f, :]       
        forecast = pca.transform(forecast_X)            
        forecast = np.sum(forecast[:, -k1:], axis=1)

        #no trade day at the end of the window (model recalibration day)
        no_trade[end_idx] = 1

        #store the forecast errors
        forecast_errors = forecast - mean_X               
        e_forecasts[end_idx:end_idx + f] = forecast_errors        

    return e_forecasts, no_trade

#load and preprocess data
data = wP.values

#drop Montepaschi
data=np.delete(data,[14],axis=1)    

#set parameters for the rolling PCA
q = 52  # Window size
f = 10  # Step size
n_rows=data.shape[0]
k1 = 10

#execute the rolling PCA function
errors, no_trade = rolling_PCA_sklearn(data[:,:], q, f, k1)

#1st plot: forecast errors
plt.plot(errors)
plt.title('Forecast errors (PLOT 4)')
plt.xlabel('Weeks')
plt.ylabel('Forecast errors')
plt.show()

#2nd plot: mean reverting strategy
pl = np.zeros(n_rows)
for i in range(q, n_rows):
    if no_trade[i] == 1:
        pl[i] = 0
    if no_trade[i] == 0:
        pl[i] = -np.sign(errors[i - 1]) * (errors[i] - errors[i - 1])


plt.plot(pl)
plt.title('Trading Strategy P&L (PLOT 5)')
plt.xlabel('Weeks')
plt.ylabel('Weekly P&L')
plt.show()


#3rd plot: cumulative P&L

sumpl = np.cumsum(pl)
plt.plot(sumpl)
plt.title('Cumulative P&L (PLOT 6)')
plt.xlabel('Weeks')
plt.ylabel('Cumulative P&L')
plt.show()





#PAIRS TRADING STRATEGY (PTS)

#PTS RANDOM FUNCTION: function to select two random stocks and plot the P&L for the mean reverting strategy
def PTS_random():
    index=np.arange(Ptest.shape[1])
    x1, x2 = random.sample(list(index), 2)
    s1,s2=Ptest[:,x1],Ptest[:,x2]

    #compute the spread
    Dtest=np.zeros(len(s1)) #D as for delta
    for i in range(1,len(s1)):
        Dtest[i]=s1[i]-s2[i]

    #mean reverting strategy
    pl=np.zeros(len(Dtest))
    for i in range(1,len(Dtest)):
        if Dtest[i-1]>=0:
            pl[i]=pl[i-1]+(Dtest[i-1]-Dtest[i])
        else:
            pl[i]=pl[i-1]+(Dtest[i]-Dtest[i-1])

    print("Banks randomly selected: ",row_index[x1],"and ",row_index[x2])

    plt.plot(pl)
    plt.title('P&L from random selection(PLOT 7)')
    plt.xlabel('Weeks')
    plt.ylabel('P&L')
    plt.show()
    
    return x1,x2,Dtest

x1,x2,Dtrain=PTS_random() #calling the function

#now, we execute the same strategy but pick pairs with KMeans clustering instead.
scaler=MaxAbsScaler()
scaled_train=scaler.fit_transform(Ptrain.T)
kmeans=KMeans(n_clusters=21,n_init=10,random_state=0).fit(scaled_train)
labels1=kmeans.labels_                      #<class 'numpy.ndarray'>
labels2=pd.DataFrame(labels1)
row_index=pd.DataFrame(row_index)
df=pd.concat([row_index,labels2],axis=1)
df.columns = ['Company', 'Cluster index']
df_sorted = df.sort_values(by='Cluster index')

#issue here is that clusters do not have all the same size
df_dup=df_sorted.index.groupby(df_sorted["Cluster index"]) #duplicates, PrettyDict type
df_dup_dict=dict(df_dup)                                   #PrettyDict to dict

#consider only clusters with 2 or more components
pairs=[]
for i in range(len(df_dup_dict)):
    if len(df_dup_dict[i])>=2:
        pairs.append(df_dup_dict[i])

pd_pairs=pd.DataFrame(pairs)





#PTS CLUSTERING FUNCTION: function to select two stocks based on clustering and plot the P&L for the mean reverting strategy
def PTS_clustering(cluster_number):
    valid_indices = pd_pairs.iloc[cluster_number, :].dropna().astype(int)
    if len(valid_indices) >= 2:
        x1, x2 = random.sample(list(valid_indices), 2)
        s1, s2 = Ptest[:, x1], Ptest[:, x2]
        
        Dtest=np.zeros(len(s1)) #D as for delta
        for i in range(1,len(s1)):
            Dtest[i]=s1[i]-s2[i]

        #mean reverting strategy
        pl=np.zeros(len(Dtest))
        for i in range(1,len(Dtest)):
            if Dtest[i-1]>=0:
                pl[i]=pl[i-1]+(Dtest[i-1]-Dtest[i])
            else:
                pl[i]=pl[i-1]+(Dtest[i]-Dtest[i-1])

        print("Banks selected by clustering: ",row_index.iloc[x1,0],"and ",row_index.iloc[x2,0])

        plt.figure(figsize=(10, 5))
        plt.plot(pl)
        plt.title('P&L from clustering (PLOT 8)')
        plt.xlabel('Weeks')
        plt.ylabel('P&L')
        plt.show()
        return x1, x2, Dtest
    
x1,x2,Dtrain=PTS_clustering(1) #calling the function
