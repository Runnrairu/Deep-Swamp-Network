# -*- coding: utf-8 -*-
import tensorflow as tf


def ResFlow(inpt,delta_t,delta_w):
    
    f_x = Res_func(inpt)
    
    
    return inpt+0.5*delta_t*f_x +0.025*delta_w*f_x
   
def Res_func(inpt):
   
   
   
   
   
   
   return output
