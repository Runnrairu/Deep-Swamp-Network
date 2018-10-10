# -*- coding: utf-8 -*-
import tensorflow as tf

def tW_def(n,name):
    if name == "Fukasawa_scheme":
        t,W = Fukasawa_scheme(1)
    elif name == "Simplicity_scheme":
        t,W = Simplicity_scheme(1)
    elif name == "Euler_Maruyama_scheme":
        t,W = Euler_Maruyama_scheme(n,1)
    elif name == "Milstein_scheme":
        t,W == Milstein_scheme(n,1)
    else:
        print("Invarid!")
    return t,W

def Fukasawa_scheme(T):
    
    return t,W

def Simplicity_scheme(T):
    
    return t,W




def Euler_Maruyama_scheme(n,T):
    
    return t,W


def Milstein_scheme(n,T):
    
    retrun t,W




def ResFlow(inpt,delta_t,delta_w):
    
    f_x = Res_func(inpt)
    
    
    return inpt+0.5*delta_t*f_x +0.025*delta_w*f_x
   
def Res_func(inpt):
    W_conv1 = weight_variable([5, 5, 64, 64])
    b_conv1 = bias_variable([64])
    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
   
   
   
   
   return output
