# DA-Power-Trading
Some strategies for short term trading in some power market


This is a project that aims to create some exploratory trading strategies based on power market fundamentals.


Notebook ImbalanceUK.ipynb 
------------------------------------
provides an analysis of the imbalance volume on an hourly granularity for the UK power market. 

================================================================================================================

PredictionStrategyUK.ipynb
------------------------------------

Provides a prediction based strategy. The prediction are made with a supervised learning classification algorithm.

================================================================================================================

MeanRevUK.ipynb
-------------------------------------

Provides two mean reverting strategies using Bollinger Bands, and RSI indicators on the imbalance volume data.

================================================================================================================

PairTrading.ipynb
--------------------------------------

Provides a pair trading strategy, further refined with more advanced techniques, by taking as pairs the imbalance volume price in UK and the spot price in UK.


**********************************CODE CONTENTS*********************************************************

AnalyticsModules
Contents:
====================

cleandata.py

--------------------

correlation.py

--------------------

devotools.py

--------------------

graphbased.py

--------------------

linear.py

=====================================================

BackTestModules
Contents:
=====================

VectorizedBase.py

---------------------

    Strategy
    SubContents:
    
    ======================
    
    StrategyZero.py
    
    ----------------------
    
    PredictionML.py
    
    ----------------------
    
    MeanReversion.py
    
    ----------------------
    
    PairTrading.py
    
    ----------------------
    
    
================================================================================================================
    
    
