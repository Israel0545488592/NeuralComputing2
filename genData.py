import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt

def_seed=None

#use defualt seed if no seed was given
def get_seed(seed=None):
    if seed == None:
        seed = def_seed
    return seed

#genData.create_A2(size=1000,rng_type="power",a=2)
#genData.create_A2(size=1000,rng_type="normal",loc=2,scale=0.1)
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


def generate_points(_min,_max,size=1000,seed=None,rng_type="",**kwargs):
    x = generate_numbers(_min,_max,size,seed=seed,rng_type=rng_type,**kwargs)
    y = generate_numbers(_min,_max,size,seed=seed+5,rng_type=rng_type,**kwargs)
    out = np.stack((x,y), axis=1)
    
    return out

#generate a DataSet for Part A
def create_A1(size=1000,seed=None):
    seed = get_seed(seed)
    out = generate_points(0,1,size,seed=seed)
    vals = np.array([np.sum(np.square(arr-0.5)) for arr in out])
    out1 = out[(vals>=0)&(vals<=0.5**2)]
    out2 = out[(vals<0)|(vals>0.5**2)]
    out = [out1,out2]
    return out

#generate a data set for Part A2
def create_A2(size=1000,seed=None,rng_type="noraml",**kwargs):
    seed = get_seed(seed)
    out = generate_points(0,1,size,seed=seed,rng_type=rng_type,**kwargs)
    vals = np.array([np.sum(np.square(arr-0.5)) for arr in out])
    out1 = out[(vals>=0)&(vals<=0.5**2)]
    out2 = out[(vals<0)|(vals>0.5**2)]
    out = [out1,out2]
    return out

#generate a data set for Part A3
def create_A3(size=1000,seed=None,**kwargs):
    seed = get_seed(seed)
    out = generate_points(-2,2,size,seed=seed,**kwargs)
    vals = np.array([np.sum(np.square(arr)) for arr in out])
    out1 = out[(vals>=2)&(vals<=4)]
    out2 = out[(vals<2)|(vals>4)]
    out = [out1,out2]
    return out



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






