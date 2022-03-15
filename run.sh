# !/bin/bash


# Including MP1,MP2,RF,SVM, and XGB

mkdir Results
mkdir Process

# cp run file into Process folder

cp MP1/model1.pkl Process/
cp MP1/model2.pkl Process/
cp MP1/MP1.py Process/
cp MP2.pkl Process/
cp RF.pkl Process/
cp SVM.pkl Process/
cp XGB.pkl Process/
cp four_model.py Process/
cp test412.csv Process/
cp test30.csv Process/
cd Process
# S30 is used here as an example. 


# clean the intermeidate files
python MP1.py test30.csv
python four_model.py test30.csv

cd ..
rm -r Process 