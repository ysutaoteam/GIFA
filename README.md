# GIFA--A method for predicting the severity of Parkinson's disease
We provide the code for the GIFA. Please follow the instructions below to use our code.
## Prerequisites
The code is tested on 64 bit Windows 10. You should also install Python 3.9 before running our code.
## UPDRS.rar files
The implementation of the GIFA method mainly includes the following steps:<br>
①GIFE;<br>
②causal feature selection IAMB; <br>
③ the combination of group interaction and target patient-specific tasks.<br>
<br>
Therefore, the motor and total folders contain the following files:
## 1）1.GIFE.py
The voice feature information of all patients is input into the GIFE and trained to obtain the shared parameters of the structure.
## 2）2.GIFA
①Use GIFE to obtain the group interaction features of target patients；<br>
②The IAMB is used to select the features that have causal relationship with the UPDRS of the target patients from the group interaction features；<br>
③Integrate the information of specific patients with the causal interaction features obtained by IAMB to improve prediction performance.
