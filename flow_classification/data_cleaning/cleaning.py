import os
import pandas as pd
import numpy as np


def separate_labels(df):
    """Returns the data without labels and the labels, separetly"""
    df_labels = df["flow"].copy()
    df = df.drop("flow", axis=1)
    return df, df_labels
    
def remove_non_log_features(df):
    """Removes the features that we are not going to use"""
    df = df.drop(["feature_1", "feature_2", "feature_3", "feature_4"], axis=1)
    return df

def feat_scaling_time(df):
    """ Feature scaling with pipeline transformers

    ## Most ML dont perform well when the input numerical values have very different scales
    # transformation ONLY fit to train set"""
    


    cat_attribs=["exp_no", "depth"]
    num_attribs= np.setxor1d(list(df.columns), cat_attribs)

    # features are standarised
    num_pipeline = Pipeline([

            ('std_scaler', StandardScaler()),
        ])


    # nothing is done to cat_attributes
    DummyScaler = FunctionTransformer(lambda x: x)

    cat_pipeline = Pipeline([
            ('dummy',DummyScaler),
    ])

    

    full_pipeline = ColumnTransformer([
            ("cat", cat_pipeline, cat_attribs),
            ("num", num_pipeline, num_attribs),
        ])


   
    return     full_pipeline

def tidy_columns(df):
        # Reorder columns
    df=test_set
    cols = df.columns.tolist()
    cols=cols[0:2]+list(["flow"])+cols[2:9]+cols[10::]
    cols
    df = df[cols] 
    return df

def average_time(df): # Average over time
    """Averages the 2D dimensional features over time. Output is a 1D arrayhas 1D features"""
    exp_no=np. unique(df["exp_no"]) # create an array with all the experiement numbers
    time_averaged=pd.DataFrame(columns=df.columns)

    for experiment in exp_no:
        exp_data=df[df["exp_no"]==experiment]

        exp_data=exp_data.set_index(['depth'])

        exp_data=exp_data.groupby('depth').mean() # averaged in time and depth
        exp_data["flow"]=np.unique(df[df["exp_no"]==experiment]["flow"])[0]
        exp_data = exp_data.reset_index(level=0)

        time_averaged=time_averaged.append(exp_data)
    return time_averaged

def augmentation(df):
    """Shifts each feature by "one pixel" backwards and forwards. We finish with 600 experiements in our data set """
    df=np.array(df)
    depth_max=26
    df_new=[]
    for exp_data in df:
        for dx in (-1,1):
            for a in range(1,8):
                new_feature=exp_data[depth_max*(a-1):depth_max*a]  
                new_feature= np.roll(new_feature, dx, axis=None)

                df_new=np.append(df_new, new_feature)
            df_new=np.append(df_new, exp_data[-1])
    print(len(df_new))
    df_new=np.append(df,df_new)
    
    df_new=df_new.reshape(120,183)
    return df_new # Augmentates data

def reshape_features(df):
    df["time"]=df["time"].astype(int)



    train_set, test_set=train_and_test_data(df) # separates into training and test set



    test_set_prepared=test_set
    train_set_prepared=train_set


    ## Reshape pandas
    column=["feature_0", "feature_1","feature_2","feature_3","feature_4","feature_5","feature_6"]

    train_set_total = pd.melt(train_set_prepared, id_vars=['flow', "exp_no","time", "depth"], value_vars=column, value_name='value')
    train_set_total = train_set_total.pivot_table("value", index=["exp_no", "flow"],  columns=[ "time","variable",'depth',])

    test_set_total = pd.melt(test_set_prepared , id_vars=['flow', "exp_no","time", "depth"], value_vars=column, value_name='value')
    test_set_total = test_set_total.pivot_table("value", index=["exp_no", "flow"],  columns=["time","variable",'depth'])

    ## remove all the times that are longer than the shortest experiement (102 seconds)
    index=np.around(np.arange(0,103), decimals=1)
    train_set_total=train_set_total[index]
    test_set_total=test_set_total[index]

    train_set_total["flow"]=train_set_total.index.get_level_values("flow")
    train_set_total.index = train_set_total.index.droplevel(1)



    test_set_total["flow"]=test_set_total.index.get_level_values("flow")
    test_set_total.index = test_set_total.index.droplevel(1)


    # Shuffle
    train_set_1D, train_labels_1D=separate_labels(train_set_total)
    shuffle_index = np.random.permutation(train_set_1D.shape[0])
    train_set_1D, train_labels_1D = train_set_1D.iloc[shuffle_index], train_labels_1D.iloc[shuffle_index]
    train_labels_1D=pd.Series.to_frame(train_labels_1D)

    test_set_1D, test_labels_1D=separate_labels(test_set_total)
    shuffle_index = np.random.permutation(test_set_1D.shape[0])
    test_set_1D, test_labels_1D = test_set_1D.iloc[shuffle_index], test_labels_1D.iloc[shuffle_index]
    test_labels_1D=pd.Series.to_frame(test_labels_1D)
    
    # change labels setupclass

    train_labels_1D = np.array(train_labels_1D["flow"])
    test_labels_1D =  np.array(test_labels_1D["flow"])
    
    return train_set_1D, train_labels_1D, test_set_1D, test_labels_1D # Reshapes pandas and drops features that have many Nans

def data_clean_1D(df, plot):
    
    data_new_features=add_new_features(df)
    data_new_features["depth"]=data_new_features["depth"]-data_new_features["depth"].min()

    data_over_time_1=average_time(data_new_features) # averages over time See appendix A
    data_over_time=data_over_time_1.drop(["feature_1", "feature_2","feature_3","feature_4"], axis=1)

    data_over_time=data_over_time.drop(["time"],axis=1) # drops time label
    train_set, test_set=train_and_test_data(data_over_time) # separates into training and test set


    # separate data and labels
    train_set, train_labels=separate_labels(train_set)
    test_set, test_labels=separate_labels(test_set)

    # apply pipeline to standarise features
    full_pipeline=feat_scaling_time(train_set)
    train_set_prepared=full_pipeline.fit_transform(train_set)  # fit and transform. We only want to do transform on data set
    test_set_prepared=full_pipeline.transform(test_set) # ONLY TRANSFORMS

    train_set_prepared  = pd.DataFrame(
            train_set_prepared ,
            dtype=float,
            columns=list(train_set.columns),
            index=train_set.index)
    train_set_prepared["flow"]=train_labels

    test_set_prepared  = pd.DataFrame(
            test_set_prepared ,
            dtype=float,
            columns=list(test_set.columns),
            index=test_set.index)
    test_set_prepared["flow"]=test_labels

    ## Reshape pandas
    column=["feature_0", "feature_1","feature_2","feature_3","feature_4","feature_5","feature_6","log_feature_1","log_feature_2","log_feature_3","log_feature_4", ]
    column=["feature_0", "feature_5","feature_6","log_feature_1","log_feature_2","log_feature_3","log_feature_4", ]

    train_set_1D = pd.melt(train_set_prepared, id_vars=['flow', "exp_no", "depth"], value_vars=column, value_name='value')
    train_set_1D = train_set_1D.pivot_table("value", index=["exp_no", "flow"],  columns=["variable",'depth'])
    train_set_1D["flow"]=train_set_1D.index.get_level_values("flow")
    train_set_1D.index = train_set_1D.index.droplevel(1)


    test_set_1D = pd.melt(test_set_prepared , id_vars=['flow', "exp_no", "depth"], value_vars=column, value_name='value')
    test_set_1D = test_set_1D.pivot_table("value", index=["exp_no", "flow"],  columns=["variable",'depth'])
    test_set_1D["flow"]=test_set_1D.index.get_level_values("flow")
    test_set_1D.index = test_set_1D.index.droplevel(1)

    # Shuffle
    train_set_1D, train_labels_1D=separate_labels(train_set_1D)
    shuffle_index = np.random.permutation(train_set_1D.shape[0])
    train_set_1D, train_labels_1D = train_set_1D.iloc[shuffle_index], train_labels_1D.iloc[shuffle_index]
    train_labels_1D=pd.Series.to_frame(train_labels_1D)

    test_set_1D, test_labels_1D=separate_labels(test_set_1D)
    shuffle_index = np.random.permutation(test_set_1D.shape[0])
    test_set_1D, test_labels_1D = test_set_1D.iloc[shuffle_index], test_labels_1D.iloc[shuffle_index]
    test_labels_1D=pd.Series.to_frame(test_labels_1D)

    # change labels setupclass

    train_labels_1D = np.array(train_labels_1D["flow"])
    test_labels_1D =  np.array(test_labels_1D["flow"])
    
    if plot==True:
        
        column_log=[ "feature_1","feature_2","feature_3","feature_4"]
        column=["log_feature_1","log_feature_2","log_feature_3","log_feature_4"]
        column=np.append( column_log, column)

        df2 = pd.melt(data_over_time_1, id_vars='flow', value_vars=column, value_name='value')
        g = sns.FacetGrid(df2, col="variable", hue="flow", sharex=False, sharey=False, col_wrap=4, legend_out=True)
        g.map(sns.kdeplot, "value", fill=True)
        g.add_legend()
    
    return train_set_1D, train_labels_1D, test_set_1D, test_labels_1D # clean data and average over time

def data_clean_aug(df):
    data_new_features=add_new_features(df)
    data_new_features["depth"]=data_new_features["depth"]-data_new_features["depth"].min()

    data_over_time_1=average_time(data_new_features) # averages over time See appendix A
    data_over_time=data_over_time_1.drop(["feature_1", "feature_2","feature_3","feature_4"], axis=1)

    data_over_time=data_over_time.drop(["time"],axis=1) # drops time label
    train_set, test_set=train_and_test_data(data_over_time) # separates into training and test set


    # separate data and labels
    train_set, train_labels=separate_labels(train_set)
    test_set, test_labels=separate_labels(test_set)

    # apply pipeline to standarise features
    full_pipeline=feat_scaling_time(train_set)
    train_set_prepared=full_pipeline.fit_transform(train_set)  # fit and transform. We only want to do transform on data set
    test_set_prepared=full_pipeline.transform(test_set) # ONLY TRANSFORMS

    train_set_prepared  = pd.DataFrame(
            train_set_prepared ,
            dtype=float,
            columns=list(train_set.columns),
            index=train_set.index)
    train_set_prepared["flow"]=train_labels

    test_set_prepared  = pd.DataFrame(
            test_set_prepared ,
            dtype=float,
            columns=list(test_set.columns),
            index=test_set.index)
    test_set_prepared["flow"]=test_labels


    ## Reshape pandas
    column=["feature_0", "feature_1","feature_2","feature_3","feature_4","feature_5","feature_6","log_feature_1","log_feature_2","log_feature_3","log_feature_4", ]
    column=["feature_0", "feature_5","feature_6","log_feature_1","log_feature_2","log_feature_3","log_feature_4", ]

    train_set_1D = pd.melt(train_set_prepared, id_vars=['flow', "exp_no", "depth"], value_vars=column, value_name='value')
    train_set_1D = train_set_1D.pivot_table("value", index=["exp_no", "flow"],  columns=["variable",'depth'])
    train_set_1D["flow"]=train_set_1D.index.get_level_values("flow")
    train_set_1D.index = train_set_1D.index.droplevel(1)


    test_set_1D = pd.melt(test_set_prepared , id_vars=['flow', "exp_no", "depth"], value_vars=column, value_name='value')
    test_set_1D = test_set_1D.pivot_table("value", index=["exp_no", "flow"],  columns=["variable",'depth'])
    test_set_1D["flow"]=test_set_1D.index.get_level_values("flow")
    test_set_1D.index = test_set_1D.index.droplevel(1)
    print(train_set_1D.shape)
    # Data Augmentation
    train_set_1D=augmentation(train_set_1D)


    # Shuffle
    train_labels_1D=train_set_1D[:,-1]
    train_set_1D= train_set_1D[:,:-1]

    shuffle_index = np.random.permutation(len(train_set_1D))
    train_set_1D, train_labels_1D= train_set_1D[shuffle_index], train_labels_1D[shuffle_index]


    test_set_1D, test_labels_1D=separate_labels(test_set_1D)
    shuffle_index = np.random.permutation(test_set_1D.shape[0])
    test_set_1D, test_labels_1D = test_set_1D.iloc[shuffle_index], test_labels_1D.iloc[shuffle_index]
    test_labels_1D=pd.Series.to_frame(test_labels_1D)

    # change labels setupclass

    #train_labels_1D = np.array(train_labels_1D["flow"])
    test_labels_1D =  np.array(test_labels_1D["flow"])
    return train_set_1D, train_labels_1D, test_set_1D, test_labels_1D # clean data and agumentates

def plot_confusion_matrix(matrix,alpha):
    """If you prefer color and a colorbar"""

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = matrix / row_sums
    np.fill_diagonal(norm_matrix, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(norm_matrix, cmap=plt.cm.gray, vmin=0, vmax=0.5)
    
    #v=np.linspace(0,0.5,num=3)
    cbar = fig.colorbar(cax)
    
    
    xaxis = np.arange(len(alpha))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(alpha)
    ax.set_yticklabels(alpha)
    plt.ylabel("Flow", size=12)
    plt.xlabel("Predicted Flow", size=12)
    return norm_matrix

