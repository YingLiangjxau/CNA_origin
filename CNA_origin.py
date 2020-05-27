# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:41:15 2019

@author: Administrator
"""

import argparse,os
import pandas as pd
import numpy as np

import autoencoder_dim
#from numba import cuda
from keras import backend as K
from sklearn.model_selection import StratifiedKFold,train_test_split
import one_D_CNN
import random

from sklearn.metrics import precision_recall_fscore_support,classification_report,recall_score,precision_score,f1_score



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #使用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #为使用CPU

def generateFeaturesAndLabels(path_gene_CNV,path_label):
    
    X_gene=pd.read_csv(path_gene_CNV,sep='\t',index_col=0)
    label_df=pd.read_csv(path_label,header=None,sep='\t',index_col=0)
    
    names=X_gene.index                  #The first column
#    cols=X_gene.columns                 #The column name
    labels=[label_df.loc[name,1] for name in names]
    X=np.array(X_gene)
    labels=np.array(labels)
    if len(X)>0:
        print("The sample feature loaded {}*{} matrix".format(len(X),len(X[0])),file=f)
    
    return X,labels
    
def l1(l):
	if isinstance(l, list):
		return l[0]
	return l

def y_posstoy_pred(y_poss):
    y_pred=[]
    for item in y_poss:
        i=np.argmax(item)
        y_pred.append(i)
    return np.array(y_pred)


def preprocessing(x_train,x_test):
    X_abs=np.abs(x_train)
    max_number=np.max(X_abs)
    x_train=x_train/(max_number)
    x_test=x_test/(max_number)
    return x_train,x_test
    

        
        
        

def mArgParsing(description='this is a description'):
    
    
    parser=argparse.ArgumentParser(description=description,epilog="The author of this program is Ying Liang, aliang1229@126.com\n")
    requiredargs=parser.add_argument_group('required arguments')
    requiredargs.add_argument('-T','--path_gene_cnv', dest="path_gene_CNV", default=None, help="Path of the gene CNV.", nargs=1, action="store", required=True, type=str)
    requiredargs.add_argument('-G','--path_label',dest="path_label",default=None,help="Path of the sample label.", nargs=1, action="store", required=True, type=str)
    parser.add_argument('-d', '--dim_number', dest="dim_number", default=100, help="The Number of Features after Dimension Reduction. ", nargs=1, action="store", type=int)
    parser.add_argument('-k', '--k_cross_validation', dest="k_cross_validation", default=10, help="k fold cross validation. ", nargs=1, action="store",type=int)
    parser.add_argument('-s', '--training_part_scale', dest="training_part_scale", default=0.1, help="Split scale for train/test. When set to 0, predictor path must be supplied. ", nargs=1, action="store", type=float)
    parser.add_argument('-o', '--output_file', dest="output_file", default=None, help="The result output path. ", nargs=1, action="store", type=str)
    
    args=parser.parse_args()
    return args

if __name__=='__main__':
    
    args=mArgParsing(description="A deep learning framework to predict tumor tissue-of-origin based on copy number alteration")
   
    
    path_gene_CNV=l1(args.path_gene_CNV)
    path_label=l1(args.path_label)
    n_features_aft=l1(args.dim_number)
    k_cross_validation=l1(args.k_cross_validation)
    training_part_scale=l1(args.training_part_scale)
    output_file=l1(args.output_file)
    if output_file is None:
        output_file=os.getcwd()
    file_name =output_file+'/result_analysis.txt'
    f= open(file_name, 'w')
       
    
    X, groups=generateFeaturesAndLabels(path_gene_CNV,path_label)
    
    set_groups=list(set(groups))
    set_groups.sort()
    digit_group=list(range(len(set_groups)))
    dict_group_digit=dict(zip(set_groups,digit_group))
    dict_digit_group=dict(zip(digit_group,set_groups))
    print(dict_digit_group,file=f)
    label_num=[dict_group_digit[l] for l in groups]
    label_num=np.array(label_num)
    n_classes=len(set(label_num))
    # X_abs=np.abs(X)
    # max_number=np.max(X_abs)
    # X=X/(max_number)
    n_features_bef=len(X[0])
    
    
    train_x,test_x,train_y,test_y=train_test_split(X,label_num,test_size=training_part_scale,random_state=random.randint(1,100))
    
    train_x,test_x=preprocessing(train_x, test_x)
    
    train_X,test_X=autoencoder_dim.autoencoder_y(train_x,n_features_bef,n_features_aft,test_x)
    
    
    
    

    

    scores,y_poss=one_D_CNN.one_D_CNN(train_x=train_X,train_y=train_y,test_x=test_X,test_y=test_y,n_features=n_features_aft,n_class=n_classes)
    f.write("The 1D CNN result of train test split:\n")
    print("The total accrucy is:",scores[1],file=f)
    y_pred=y_posstoy_pred(y_poss)
 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    class_rep=classification_report(test_y,y_pred)
    print(class_rep,file=f)
    recall_micro=recall_score(test_y,y_pred,average='micro')
    print("The micro recall of 1D CNN is:",recall_micro,file=f)
    precision_micro=precision_score(test_y,y_pred,average='micro')
    print("The micro precision of 1D CNN is:",precision_micro,file=f)
    f1_micro=f1_score(test_y,y_pred,average='micro')
    print("The micro f1 of 1D CNN is:",f1_micro,file=f)
    
    
    K.clear_session()
    
    
    




    
    
    ######## K fold training and test    ########
    print("########The results of k fold validation###########",file=f)
    print('\n\n\n',file=f)
    kfold=StratifiedKFold(n_splits=k_cross_validation,random_state=random.randint(1,100),shuffle=True)
    cnn_cvscore=[]
 

    cnn_pre_total=np.zeros(n_classes)
    cnn_rec_total=np.zeros(n_classes)
    cnn_fsc_total=np.zeros(n_classes)

    for train, validation in kfold.split(X,label_num):
        K.clear_session()
        ######### CNN classification ##########
        x_p_train,x_p_test=preprocessing(X[train],X[validation])
        k_train_X,k_test_X=autoencoder_dim.autoencoder_y(x_p_train,n_features_bef,n_features_aft,x_p_test)
        scores,y_poss=one_D_CNN.one_D_CNN(train_x=k_train_X,train_y=label_num[train],test_x=k_test_X,test_y=label_num[validation],n_features=n_features_aft,n_class=n_classes)
        cnn_cvscore.append(scores[1]*100)
        y_pred=y_posstoy_pred(y_poss)
        cnn_precision,cnn_recall,cnn_fscore,_=precision_recall_fscore_support(label_num[validation],y_pred)
        cnn_pre_total=cnn_pre_total+cnn_precision
        cnn_rec_total=cnn_rec_total+cnn_recall
        cnn_fsc_total=cnn_fsc_total+cnn_fscore
        print('The precision of one time cnn classification is :',cnn_precision,file=f)
        print('The recall of one time cnn classification is :',cnn_recall,file=f)
        print('The fscore of one time cnn classification is :',cnn_fscore,file=f)
        
  
       

        
        
    cnn_pre_total=cnn_pre_total/k_cross_validation
    cnn_rec_total=cnn_rec_total/k_cross_validation
    cnn_fsc_total=cnn_fsc_total/k_cross_validation

    
    print('The precision of cnn classification is :',cnn_pre_total,file=f)
    print('The recall of cnn classification is :',cnn_rec_total,file=f)
    print('The fscore of cnn classification is :',cnn_fsc_total,file=f)
    print('The accuracy of cnn classification of k fold validation is:',cnn_cvscore,file=f)

    
    


    
    
    f.close()
    
    
    
    
    
        
            
             
            
            
    
    
    
