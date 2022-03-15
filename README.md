# MP-model

# Application of machine learning with RF-RFE in clinical trials for noninvasive prenatal testing of fetal trisomy 13, trisomy 18 and trisomy 21

This programs run on Linux system.

1.Dependencies

  python==3.8.12
  numpy==1.21.2
  panda==1.3.4
  sklearn==1.0.1
  joblib==1.1.0
  mlxtend==0.19.0
  xgboost==1.4.2

2. Examples

There are two sample datasets. File_name is test30.csv or test412.csv. This paper includes five models for detecting fetal aneuploid.

MP1 is the best predictor of the five models. An example of MP1 would be:

python MP1.py File_name

If you want to get all the results for the five models, use the following code:

./run.sh


For any questions, please contact us by xiaohansun@bjut.edu.cn.
