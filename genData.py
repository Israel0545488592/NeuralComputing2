import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt
from sympy import true

def_seed=None

#use defualt seed if no seed was given
def get_seed(seed=None):
    if seed == None:
        seed = def_seed
    return seed


###--------------------###
### number generators
###--------------------###

#generate random numbers and allow usage of weighted random numbers, using either power or normal distribution
#args for power are: a
#args for normal are: loc, scale
#note, power will generate more numbers that are closer to both min and max instead of just max like normaly
def generate_numbers(_min,_max,size=1000,seed=None,rng_type="",**kwargs):
    seed = get_seed(seed)
    
    rng = np.random.RandomState(seed)
    if rng_type == "normal":
        rnd = rng.normal(size=size,**kwargs)
        rnd = -min(0,np.min(rnd)) + rnd
        rnd = _max*rnd/np.max(rnd)
        
    elif rng_type == "power":
        rnd = np.abs((_max-_min)*rng.power(size=size,**kwargs) + _min + rng.choice([0,-1],size))

    else:
        rnd = rng.uniform(_min,_max,size=size)
    return rnd


def generate_points(_min,_max,decision_func=None,size=1000,seed=None,rng_type="",**kwargs):
    curr_size = 0
    out = []
    i = 0
    while True:
        #generate random x and y values, make sure the seed is unique yet reproducable for each call
        x = generate_numbers(_min,_max,size-curr_size//(i+1),seed=2*(seed+i),rng_type=rng_type,**kwargs)
        y = generate_numbers(_min,_max,size-curr_size//(i+1),seed=2*(seed+i)+1,rng_type=rng_type,**kwargs)
        temp = np.stack((x,y), axis=1)

        if decision_func != None:
            #resize temp and make it keep only the good values
            temp = temp[decision_func(temp)]
        temp = temp[:size-curr_size]

        #add the correct values to the final array 
        if len(out)==0:
            out = temp
        else:
            out = np.concatenate((out,temp), axis=0)
        
        curr_size = out.shape[0]
        i+=1
        #once we reached our target size, stop
        if curr_size == size:
            break
        

    return out


###--------------------###
### decision functions
###--------------------###

#decision funcion that returns what elements we generated and we want to keep for part A 1 and 2
def A_1_2_decision_func(arr):
    vals = np.array([np.sum(np.square(e-0.5)) for e in arr])
    vals[(vals>=0)&(vals<=0.5**2)] = 0
    
    return vals == 0

#decision funcion that returns what elements we generated and we want to keep for part A 3
def A_3_decision_func(arr):
    vals = np.array([np.sum(np.square(e)) for e in arr])
    vals[(vals>=2)&(vals<=4)] = 0
    return vals == 0




###--------------------###
### dataset creation
###--------------------###

#generate a DataSet for Part A1
def create_A1(size=1000,seed=None):
    seed = get_seed(seed)
    out = generate_points(0,1,A_1_2_decision_func,size=size,seed=seed)
    return out

#generate a data set for Part A2
def create_A2(size=1000,seed=None,rng_type="noraml",**kwargs):
    seed = get_seed(seed)
    out = generate_points(0,1,A_1_2_decision_func,size,seed=seed,rng_type=rng_type,**kwargs)
    return out

#generate a data set for Part A3
def create_A3(size=1000,seed=None,**kwargs):
    seed = get_seed(seed)
    out = generate_points(-2,2,A_3_decision_func,size,seed=seed,**kwargs)
    return out


###--------------------###
### helper functions
###--------------------###

#helper function to plot model results
def plot_model_results(y_true,y_pred,labels=None,name=None):
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cm = ConfusionMatrixDisplay.from_predictions(y_true,y_pred,display_labels=labels,values_format="d",ax=ax1)
    
    fig.set_size_inches(20,10)
    num = cm.text_[0][0].get_text()
    txt = "True Negative"+ "\n" + num + "\n" + '{0:.3f}'.format(100*(int(num)/len(y_pred)))+"%"
    cm.text_[0][0].set_text(txt)
    
    num = cm.text_[1][0].get_text()
    txt = "False Negative"+ "\n" + num + "\n" + '{0:.3f}'.format(100*(int(num)/len(y_pred)))+"%"
    cm.text_[1][0].set_text(txt)

    num = cm.text_[0][1].get_text()
    txt = "False Positive"+ "\n" + num + "\n" + '{0:.3f}'.format(100*(int(num)/len(y_pred)))+"%"
    cm.text_[0][1].set_text(txt)

    num = cm.text_[1][1].get_text()
    txt = "True Positive"+ "\n" + num + "\n" + '{0:.3f}'.format(100*(int(num)/len(y_pred)))+"%"
    cm.text_[1][1].set_text(txt)
    
    RocCurveDisplay.from_predictions(y_true,np.random.choice([0,1],size=len(y_true)),ax=ax2,linestyle='--',name="Random classifier")
    
    
    #ax2.plot(falsePosRate, truePosRate, linestyle='--',label="random classifier")
    RocCurveDisplay.from_predictions(y_true,y_pred,ax=ax2)
    if name != None:
        plt.savefig(name)