# -*- coding: utf-8 -*-
import tensorflow as tf


def ResFlow(inpt,delta_t,delta_w):
    
    f_x = Res_func(inpt)
    
    
    return inpt+0.5*delta_t*f_x +0.025*delta_w*f_x
   
def Res_func(inpt):
    W_conv1 = weight_variable([5, 5, 64, 64])
    b_conv1 = bias_variable([64])
    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
   
   
   
   
   return output
