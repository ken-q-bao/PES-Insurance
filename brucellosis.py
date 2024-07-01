# -*- coding: utf-8 -*-
"""
Created on Tuesday 7/15/2023

@author: Ken
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.optimize as opt

##############################################################################
######################## Set Working Directory ###############################
##############################################################################
os.chdir('C:\\Users\\kqtqh\\Desktop\\KensWork\\UCSB2\\PES_insur\\simulations')

##############################################################################
######################## Parameter Value Designation #########################
##############################################################################
high_prob = .18                                 # prob of bad state under WFF
low_prob = .001                                 # prob of bad state under traditional fencing
  
wealth = 400*1.8*1150*.32*.98    # Wealth = net income 400 cattle, $1.8/lb, 1150 lbs/cattle, 32% margin, 98% survival rate
ushock = 140000                  # e shock value = cost of mandatory quarantine procedures
lshock = 0                       # no cost if no outbreak (normalization)

cost = ((43560000)**.5)*4*1.48*.6 # WFF cost = 2.5 acres per head with 400 heads and fencing the whole lot means 1000 acres = 43.56 million sq ft
lamduh = np.array([2,3,4,5,6])    # initialize values for coef of RRA


insure = np.linspace(0,1,num=11)   # create various points I to evaluate over
price = 70000                      # only used for initializing the numeric optimization


pol_star = np.zeros(shape = (len(lamduh),len(insure))) # initialize the output dataframe

##############################################################################
######################## Setting up Custom Functions #########################
##############################################################################
# Define utility
def util(c, lamduh): 
    if c>0:
        u = (100000000000/(1-lamduh))*(c**(1-lamduh)) # need large coef since wealth is high, root search tolerance will be arbitrarily satsified
        
    else:
        u = -9999999999999
    
    return u

# Define function to root search over
def root(p, I, risk):
    a = wealth + p - cost - ushock*(1-I)
    b = wealth + p - cost - lshock*(1-I)
    G = high_prob*util(a, risk) + (1-high_prob)*util(b,risk) - low_prob*util(wealth-ushock, risk) - (1-low_prob)*util(wealth-lshock, risk)
    return G


##############################################################################
####################### Root Search for Optimal p(I) #########################
##############################################################################
for j in range(len(lamduh)):        # for each coef of RRA
    for i in range(len(insure)):    # and each value of I
        risk = lamduh[j]
        I = insure[i]
        result = opt.root_scalar(root,
                                 args = (I, risk),
                                 bracket = (-price, wealth), # bisection method requires opposite signs in f(a) and f(b)
                                 maxiter = 10000,
                                 xtol = 10e-20,
                                 method = 'bisect')
        pol_star[j,i] = result.root


##############################################################################
############################# Graphing Results ###############################
##############################################################################
plt.plot(insure*100,pol_star[0,:]/1000, color='red', label = 'Coefficient RRA = 2', linestyle='solid')
plt.plot(insure*100,pol_star[1,:]/1000, color='blue', label = 'Coefficient RRA = 3', linestyle = 'dashed')
plt.plot(insure*100,pol_star[2,:]/1000, color='green', label = 'Coefficient RRA = 4', linestyle = 'dotted')
plt.plot(insure*100,pol_star[3,:]/1000, color='purple', label = 'Coefficient RRA = 5', linestyle = (0, (3, 1, 1, 1, 1, 1)))
plt.plot(insure*100,pol_star[4,:]/1000, color='brown', label = 'Coefficient RRA = 6', linestyle = (5, (10, 3)))
plt.axhline(cost/1000, color = 'black', label = 'WFF Cost', linestyle = 'dashdot')
plt.xlabel("Indemnity Rate (%)")
plt.ylabel("Pay Rate (1000's)")
plt.legend()
plt.savefig('cost savings full indemnity - cattle.jpg',bbox_inches = 'tight', dpi=150)
plt.show()


##############################################################################
######################### Calculations for Table 1 ###########################
##############################################################################
plotdat = {'p0 per g': pol_star[:,0]/cost, 'p1 per g': pol_star[:,10]/cost, 'p0':pol_star[:,0], 'p1':pol_star[:,10]}
plotdat = pd.DataFrame(plotdat)
plotdat['savings'] = (plotdat['p0'] - plotdat['p1']-(high_prob*ushock))/plotdat['p0']
