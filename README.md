# fingerPrintDiff
This project is just for our deep learning course design.

# Dependence
python3.6
keras==2.0.9
tensorflow==1.2.0
numpy==1.19.3

# Abstract
This project uses the Sokoto Coventry Fingerprint Dataset (SOCOFing),
a biometric fingerprint database designed for academic research purposes. 
SOCOFing is made up of 6,000 fingerprint images from 600 African subjects. 
SOCOFing contains unique attributes such as labels for gender, hand and finger 
name as well as synthetically altered versions with three different levels of 
alteration for obliteration, central rotation, and z-cut. The dataset is freely available
for noncommercial research purposes at: https://www.kaggle.com/ruizgara/socofing

# File description
gen_data.py:generate data Sequence.
Network.py:define the deep neural network.
sourceCode.py:this code come from https://my.oschina.net/u/4581492/blog/4520918. However,we think there is something wrong with it.
main.py:train and test the gender recognition model and test it.
try.py:train and test the five finger(thumb,index,middle,ring,little) recognition model and test it.
thumb.py:train and test the one of the five finger(thumb,index,middle,ring,little) recognition model and test it.

# Usage
## dataSet
    Directory structure:
        --test_Altered-Easy
        --test_Altered-Hard
        --test_Altered-Medium
        --test_Real
        --train_Altered-Easy
        --train_Altered-Hard
        --train_Altered-Medium
        --train_Real
        --val_Altered-Easy
        --val_Altered-Hard
        --val_Altered-Medium
        --val_Real
     note:train_**** include ID(1-449)
          va_l**** include ID(450-500)  
          test_**** include ID(501-600)
       
  ## run
    main.py:modify the data_dir to fit yours,train the model if you input train else test.
    try.py:modify the data_dir to fit yours,train the model if you input train else test.
    thumb.py:modify the data_dir to fit yours,train the model if you input train else test and you can simply modify the mode='middle','ring','index','litte'. 



# train and val dataSet
    total samples: 45997, training samples: 41289, validation samples: 4708
    mean 0.5697416079349641
    std 0.3744768065561571
    
    
# Result
    We can't use fingerprint images to identify gender, but we can do a good job of telling whether a fingerprint image is a thumb or not.



# References
[1]Shehu, Yahaya & Ruiz-Garcia, Ariel & Palade, Vasile & James, Anne. (2018). Sokoto Coventry Fingerprint Dataset. 
