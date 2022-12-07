# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:39:08 2022

@author: Ioannis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import keras
import time
# from google.colab import files
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense

def plot_2D(TestingData,
            tr_history,
            eps,
            bat,
            PD,
            neur_list,
            activation_func_list,
            layers,
            extra):
    
    units = {
    'M'  :'M_{\odot}',
    'R'  :'km',
    'p_c':'10^{25} dyn/cm^2',
    'a'  :'km^2'}
    
    if extra:
        x = TestingData.p_c.values
        
        y1 = TestingData.M.values
        y2 = TestingData.M_pred.values
        
        z1 = TestingData.R.values
        z2 = TestingData.R_pred.values

        fig, ax = plt.subplots(1, 2, figsize=(18,5))
        ax[0].scatter(x,y1, color = 'c', label = 'True')
        ax[0].scatter(x,y2, color = 'm', label = 'Predicted')
        ax[0].legend()
        ax[0].set_title("Mass fitting")
        
        ax[1].scatter(x,z1, color = 'c', label = 'True')
        ax[1].scatter(x,z2, color = 'm', label = 'Predicted')
        ax[1].legend()
        ax[1].set_title("Radius fitting")

        # Naming
        name = "$p_c$ [$"+units["p_c"]+"$]"
        ax[0].set_xlabel(name)
        name = "$M$ [$"+units["M"]+"$]"
        ax[0].set_ylabel(name)

        name = "$p_c$ [$"+units["p_c"]+"$]"
        ax[1].set_xlabel(name)
        name = "$R$ [$"+units["R"]+"$]"
        ax[1].set_ylabel(name)

        fig.suptitle("Extra plot")

        plt.show()
    else:
        x1 = 10**TestingData.R.values
        y1 = 10**TestingData.M.values
    
        x2 = 10**TestingData.R_pred.values
        y2 = 10**TestingData.M_pred.values
    
        fig, ax = plt.subplots(1, 3, figsize=(18,5))
        ax[0].scatter(x1,y1, color = 'c', label = 'True')
        ax[0].scatter(x2,y2, color = 'm', label = 'Predicted')
        ax[0].legend()
        ax[0].set_xlabel('$ R \ [km] $')
        ax[0].set_ylabel('$ M \ [M_{\odot}] $')
        ax[0].set_title("Testing Data")
    
    
        val_loss_list = tr_history['val_loss']
        loss_list     = tr_history['loss']
        epos          = np.linspace(1,eps,eps)
        ax[1].semilogy(epos, loss_list    , label = 'Training loss')
        ax[1].semilogy(epos, val_loss_list, label = 'Validation loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss per epoch')
        ax[1].legend()
        ax[1].set_title("Mean Square Error")
    
    
    
    
        ax[2].semilogy(PD, label = 'Testing set')
        ax[2].semilogy((10**(-1))*np.ones(len(PD)), '--')
        ax[2].semilogy((10**(-2))*np.ones(len(PD)), '--')
        ax[2].set_xlabel('Points')
        ax[2].set_ylabel('Percentage Distance (%)')
        ax[2].legend()
        ax[2].set_title("Percentage Distance")
    
        # Naming
        name_neur = str(neur_list[0]) + " "
        name_act  = str(activation_func_list[0]) + " "
        for indx in range(1,layers):
            name_neur = name_neur + str(neur_list[indx]) + " "
            name_act  = name_act  + str(activation_func_list[indx]) + " "
    
        name_bat  = str(bat)
        name_eps  = str(eps)
    
        fig.suptitle("Neurons: "+name_neur+
                  " / Activation functions: "+name_act+
                  " / Batch: "+name_bat+
                  " / Epochs:"+name_eps)
    
        plt.show()
    return

"""
def plot_color( TestingData,
                tr_history,
                eps,
                bat,
                PD,
                neur_list,
                activation_func_list,
                layers,
                extra,
                every):
    
    alpha_list = TestingData.a.values
    a = list(alpha_list)
    
    n_lines = len(a)
    
    clrs = a.copy()
    clrs.sort()
    
    norm = mpl.colors.Normalize(vmin=min(a), vmax=max(a))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)
    cmap.set_array([])
    
    units = {
    'M'  :'M_{\odot}',
    'R'  :'km',
    'p_c':'10^{25} dyn/cm^2',
    'a'  :'km^2'}
    
    if extra:
        x = TestingData.p_c.values
        
        y1 = TestingData.M.values
        y2 = TestingData.M_pred.values
        
        z1 = TestingData.R.values
        z2 = TestingData.R_pred.values

        fig, ax = plt.subplots(1, 2, figsize=(18,5))
        ax[0].scatter(x,y1, color = 'c', label = 'True')
        ax[0].scatter(x,y2, color = 'm', label = 'Predicted')
        ax[0].legend()
        ax[0].set_title("Mass fitting")
        
        ax[1].scatter(x,z1, color = 'c', label = 'True')
        ax[1].scatter(x,z2, color = 'm', label = 'Predicted')
        ax[1].legend()
        ax[1].set_title("Radius fitting")

        # Naming
        name = "$p_c$ [$"+units["p_c"]+"$]"
        ax[0].set_xlabel(name)
        name = "$M$ [$"+units["M"]+"$]"
        ax[0].set_ylabel(name)

        name = "$p_c$ [$"+units["p_c"]+"$]"
        ax[1].set_xlabel(name)
        name = "$R$ [$"+units["R"]+"$]"
        ax[1].set_ylabel(name)

        fig.suptitle("Extra plot")

        plt.show()
    else:
        x1 = 10**TestingData.R.values
        y1 = 10**TestingData.M.values
    
        x2 = 10**TestingData.R_pred.values
        y2 = 10**TestingData.M_pred.values
    
        fig, ax = plt.subplots(1, 3, figsize=(18,5))
        ax[0].scatter(x1,y1, color = 'c', label = 'True')
        ax[0].scatter(x2,y2, color = 'm', label = 'Predicted')
        ax[0].legend()
        ax[0].set_xlabel('$ R \ [km] $')
        ax[0].set_ylabel('$ M \ [M_{\odot}] $')
        ax[0].set_title("Testing Data")
    
    
        val_loss_list = tr_history['val_loss']
        loss_list     = tr_history['loss']
        epos          = np.linspace(1,eps,eps)
        ax[1].semilogy(epos, loss_list    , label = 'Training loss')
        ax[1].semilogy(epos, val_loss_list, label = 'Validation loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss per epoch')
        ax[1].legend()
        ax[1].set_title("Mean Square Error")
    
    
    
    
        ax[2].semilogy(PD, label = 'Testing set')
        ax[2].semilogy((10**(-1))*np.ones(len(PD)), '--')
        ax[2].semilogy((10**(-2))*np.ones(len(PD)), '--')
        ax[2].set_xlabel('Points')
        ax[2].set_ylabel('Percentage Distance (%)')
        ax[2].legend()
        ax[2].set_title("Percentage Distance")
    
        # Naming
        name_neur = str(neur_list[0]) + " "
        name_act  = str(activation_func_list[0]) + " "
        for indx in range(1,layers):
            name_neur = name_neur + str(neur_list[indx]) + " "
            name_act  = name_act  + str(activation_func_list[indx]) + " "
    
        name_bat  = str(bat)
        name_eps  = str(eps)
    
        fig.suptitle("Neurons: "+name_neur+
                  " / Activation functions: "+name_act+
                  " / Batch: "+name_bat+
                  " / Epochs:"+name_eps)
    
        plt.show()
    return

"""
    
    
    
    



def logdata(data, log_vars, lin_vars):
    """

    Parameters
    ----------
    data : DataFrame
            Linear data.
    log_vars : List of strings 
            Variables to log.
    lin_vars : List of strings
            Variables to keep linear.

    Returns
    -------
    datalog : DataFrame
            Appropriate data for preparation.

    """
    
    datalog = pd.DataFrame({})
    
    for var in log_vars:
        datalog[var] = np.log10(data[var].values)
    
    for var in lin_vars:
        datalog[var] = data[var].values
        
    return datalog




def prepare(data,
            Predictors,
            scale,
            talk):
    """
    

    Parameters
    ----------
    data : DataFrame
        Data to be prepared for ANN.

    Returns
    -------
    X_train : array
        Input variables of Training set.
    X_test  : array
        Input variables of Testing set.
    y_train : array
        Output variables of Training set.
    y_test  : array
        Output variables of Testing set
    PredictorScalerFit : StandardScaler
        Predictor Scaler.
    TargetVarScalerFit : StandardScaler
        Target Variable Scaler.

    """
    data = data.sample(frac=1).reset_index(drop=True)
    
    TargetVariable=['M','R']
    
    X=data[Predictors].values
    y=data[TargetVariable].values
    
    if scale:
        from sklearn.preprocessing import StandardScaler
        PredictorScaler=StandardScaler()
        TargetVarScaler=StandardScaler()
    
        PredictorScalerFit=PredictorScaler.fit(X)
        TargetVarScalerFit=TargetVarScaler.fit(y)
    
        X=PredictorScalerFit.transform(X)
        y=TargetVarScalerFit.transform(y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 3)
    
    if talk:
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
    return X_train, X_test, y_train, y_test, PredictorScalerFit, TargetVarScalerFit, data



def ann(eps,
        bat,
        layers,
        X_train,
        y_train,
        X_test,
        y_test,
        neur_list,
        activation_func_list,
        timer,
        performance,
        show_plot,
        show_extra_plot,
        PredictorScalerFit,
        TargetVarScalerFit,
        Predictors,
        optimizer):

  
    # Check
    if (len(activation_func_list) != layers) or (len(neur_list) != layers):
        return "Structure error"
    
    # Create the model
    model = Sequential()
    
    # Model's structure
    model.add(Dense(units = neur_list[0], input_dim = X_train.shape[1], kernel_initializer = 'normal', activation = activation_func_list[0]))
    for lay in range(1,layers):
        model.add(Dense(units = neur_list[lay], kernel_initializer = 'normal', activation = activation_func_list[lay]))
    model.add(Dense(2, kernel_initializer='normal'))

    # Compile
    model.compile(loss='mean_squared_error', optimizer = optimizer)

    # Fit Training set
    if timer:
        start_time = time.time()
        trial      = model.fit(X_train, y_train, batch_size = bat, epochs = eps, verbose=0, validation_split = 0.2)
        end_time   = time.time()
        exe_time   = end_time - start_time
    else:
        trial      = model.fit(X_train, y_train, batch_size = bat, epochs = eps, verbose=0, validation_split = 0.2)

    # Predict Testing set
    Predictions = model.predict(X_test)
    Predictions = TargetVarScalerFit.inverse_transform(Predictions)
    y_test_orig = TargetVarScalerFit.inverse_transform(y_test)
    Test_Data   = PredictorScalerFit.inverse_transform(X_test)

    TestingData = pd.DataFrame(data=Test_Data, columns=Predictors)

    TestingData['M'] = y_test_orig.T[0]
    TestingData['R'] = y_test_orig.T[1]

    TestingData['M_pred'] = Predictions.T[0]
    TestingData['R_pred'] = Predictions.T[1]

    # Appropriate norms
    Testing_norm   = np.sqrt((10**TestingData.M.values)**2 + (10**TestingData.R.values)**2)
    # Predicted_norm = np.sqrt((10**TestingData.M_pred.values)**2 + (10**TestingData.R_pred.values)**2)
    Test_pred_norm = np.sqrt((10**TestingData.M.values-10**TestingData.M_pred.values)**2 + (10**TestingData.R.values-10**TestingData.R_pred.values)**2)

    # Percentage distance
    PD = 100*(Test_pred_norm/Testing_norm)

    # MAPE_R
    mape_R = 100*np.mean(np.abs((10**TestingData.R.values - 10**TestingData.R_pred.values)/(10**TestingData.R.values)))

    # MAPE_M
    mape_M = 100*np.mean(np.abs((10**TestingData.M.values - 10**TestingData.M_pred.values)/(10**TestingData.M.values)))

    # MAPE
    mape = (mape_R + mape_M)/2

    # Printing
    print('MPD   : ', round(np.mean(PD), 6), ' %')
    print('MAPE  : ', round(mape,6), ' %')

    performance_dic = {}
    if performance:
        performance_dic = {
            'MPD'           : [np.mean(PD)],
            'MAPE_R'        : [mape_R],
            'MAPE_M'        : [mape_M],
            'MAPE'          : [mape],
            'Max_PD'        : [max(PD)],
            'Fin_Val_loss'  : [trial.history['val_loss'][-1]],
            'Fin_Train_loss': [trial.history['loss'][-1]],
            'Min_Val_loss'  : [[min(trial.history['val_loss']),list(trial.history['val_loss']).index(min(trial.history['val_loss']))]],
            'Min_Train_loss': [[min(trial.history['loss']),list(trial.history['loss']).index(min(trial.history['loss']))]]
            }

    if timer:
        performance_dic['Exe_time'] = [exe_time]

    performance_dic['eps']                  = [eps]
    performance_dic['bat']                  = [bat]
    performance_dic['neur_list']            = [neur_list]
    performance_dic['activation_func_list'] = [activation_func_list]

    # Plotting
    if X_train.shape[1] == 1:
        if show_plot:
            plot_2D(TestingData          = TestingData,
                    tr_history           = trial.history,
                    eps                  = eps,
                    bat                  = bat,
                    PD                   = PD,
                    neur_list            = neur_list,
                    activation_func_list = activation_func_list,
                    layers               = layers,
                    extra                = False)
            
        if show_extra_plot:
            plot_2D(TestingData          = TestingData,
                    tr_history           = trial.history,
                    eps                  = eps,
                    bat                  = bat,
                    PD                   = PD,
                    neur_list            = neur_list,
                    activation_func_list = activation_func_list,
                    layers               = layers,
                    extra                = True)
    # elif X_train.shape[1] == 2:
        
    
    return performance_dic, TestingData, model


    


