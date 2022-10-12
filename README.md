# MultiCFS--A method for predicting the severity of Parkinson's disease
We provide the code for the MultiCFS. Please follow the instructions below to use our code.
## Prerequisites
The code is tested on 64 bit Windows 10. You should also install Python 3.9 before running our code.
## motor-UPDRS.rar and total-UPDRS.rar files
The implementation of the MultiCFS method mainly includes the following steps:<br>
①Multi-task bidirectional LSTM;<br>
②causal feature selection IAMB; <br>
③ the combination of MTL and target patient-specific tasks.<br>
<br>
Therefore, the motor and total folders contain the following files:
## 1）1.Multit-task Bidirectional Long Short-Term Memory(M-BiLSTM).py
The voice feature information of all patients is input into the M-BiLSTM and trained to obtain the shared parameters of the structure.
## 2）2.MultiCFS
①Use M-BiLSTM to obtain the shared parameter features of target patients；<br>
②The IAMB is used to select the features that have causal relationship with the UPDRS of the target patients from the shared parameter features；<br>
③Integrate the information of specific patients with the local causal features obtained by IAMB to improve prediction performance.
