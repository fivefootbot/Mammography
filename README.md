BREAST CANCER PREDICTION  

  

Breanna Moore, Amy McCaughan, Kenny VanGemert  

  

1. Problem Statement  

   

The problem we are trying to solve is detecting breast cancer from screening mammogram images. We want a low F1 score meaning that we were accurate in classifying breast cancer in patients who had it and did not have too many false positives or negatives. Having too many false positives or false negatives is bad for this problem because that can have drastic emotional and financial consequences on patients. The data consists of mammogram images taken during regular screenings that patients have.  

  

2. Data and Exploratory Analysis  

 

We got the data from Kaggle (https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data) where the mammograms are in dicom format. There are normally 4 images per patient, but that is not always the case in the data.  

   

The data we are using is extremely large, the data is extremely clean with only the BIRADS column showing missing information. Based on this and the rest of the data, it can be assumed there was no reason for a follow up with that patient and that they do not have breast cancer.  

 

One other thing about this data is that the pictures are actual mammography's stored in a dcm file, which is what mammograms are stored in. These files are what is extraordinarily large, too large for us to work with directly without much stronger hardware. Therefore, the size of the images must be slimmed down, they currently range anywhere from (~2000, ~2000) to (~5500, ~2000), therefore they will need to drop in size to around (128, 128) for us to comfortably work with them.  

 

There are no outliers or impurities in the data set. We used the pydicom, os, and matplotlib libraries to print the images to the screen to better see what the images were showing. We used the TN Tech OnDemand access to the cluster to get the data on our computers. This is because the data is 314GB and we could not download it on our personal computers. Through OnDemand we can see the data in a shared folder as well as each of us have a personal folder to connect to Jupyter Notebook in and write the code for our models.  

 

We also have Git to push and pull the data and our models too so that we can stay organized. Other libraries that are used are pandas for working with the .csv files, torch for using the gpu and creating a tensor out of the data, NumPy for working with the data and sklearn and its different modules for the svm and results. 

 

Below are a few images showcasing how the mammogram images appear. Some have been pre-labeled stating the view of the breast directly on the image, whereas most of them do not, showing just the breast alone. 

 

As can be seen, the mammogram images that are of patient 10011 have been pre-labeled directly on the image, showing R for right, and MLO for mediolateral oblique view. Most images do not contain this; however, it is worth noting that some do. 

 

3. Methods  

  

The methods we are using are an SVM and if time allows, CNN as well. We chose these methods because we are dealing with many images and these models handle them well. For this same reason we ruled out using a regression model since we do not have only numeric data.  

 

The SVM will allow us to use a bit of a simpler model at first to almost use it as a baseline for determining how necessary the CNN may be. If the SVM does not turn out to be a smart choice, even after tuning, or the SVM provides satisfactory results and we want to see if there are better options, then a CNN will be used.  

 

We discovered that an SVM would not work well for our data because the other variables like machine id, and views were not working great with this type of model. In a neural network, these get assigned weights based on importance, however for the SVM, it would try and use those variables as a predictor. These variables, particularly the views, are somewhat important when looking at breast cancer since one breast may contain cancer whilst the other does not, meaning the SVM would not have a great time working with this data. 

 

CNN is one of the best choices for this data since instead of having to work around different variables like BIRADS, machine id, views, etc., we can just feed all the raw data into the CNN and have it learn on cancer recognition. This makes it not only more simplistic in terms of the work needing to be done, but also faster since we do not need to worry about separating out data and removing variables since the CNN will do that for us by applying the weights to the variables, choosing what is most important and what is not.  

 

CNNs are also one of the best choices when it comes to working with images, so to not use it would hinder the potential performance of any model that we create, since a CNN in this instance will tend to always be more accurate than any other model. Finally, a CNN is also a great choice because due to the potential accuracy that we can get, having a model that can truly understand and learn what it is we are training it on is the important thing, this may not be possible to do with say an SVM since the CNN can assign importance with more accuracy. 

 

GAN (generative adversarial network) could also be used to predict if a breast has cancer or not. GAN works by creating more synthetic data to train on so that it can get a high accuracy without overfitting the data. GAN consists of a generator and a discriminator. The generator learns to generate realistic-looking images of breast cancer tissue, while the discriminator learns to distinguish between real and synthetic images.  


 

4. Tools  

  

To download and view the entire dataset, we utilized Tennessee Tech’s Open OnDemand web interface to access the high-performance computing services provided to us. We are using this because this was where we were able to store all our data to utilize it through Jupyter notebook with python which is embedded in the OnDemand site.  

 

We installed the libraries pandas, TensorFlow, and NumPy for the code so that we could run the models. We tried using deepnote but we were only able to download 5GB of the 314GB of data into the notebook.  

 

6. Results 

 

When looking at the metrics, it is important to note the model was given 3 different thresholds to calculate at, however, due to some weird error, only 0.5 produced any real result, so for the calculations, they will all be based off a threshold of 0.5.  

 

We got a precision of 0.3062, a recall of 0.3621, F1 score of 0.3314, and an accuracy of 0.5018. On the test data the model predicted 0.49373 for the left breast, and 0.49372 for the right breast. These scores are exceptionally good and way higher than what was anticipated. For this problem we wanted a high F1 score because the context of the problem can result in life-or-death scenarios. With a high F1 score, we can ensure that the model is the best at predicting correctly without too many misclassifications.  

 

When looking at our F1 score, we see it is a bit on the lower side, this is still somewhat ok in our scenario since we were not able to tune the model past what the parameters are set at, potentially resulting in lower metrics all around. However, even though our metrics fell a little bit on the lower side, the model still had good prediction values and had one of the higher scores in the competition. Even though the competition was not active at the time of the assignment, if it were to have been a submission in the competition, we would have been placed in the top 15, this is why we were somewhat surprised at the score the model was achieving.  


 

As stated in the first paragraph, we encountered some weird errors with the thresholds for the metrics, below we can see this. When looking at the recall for training recall at 0.4 and 0.6, we see a spike, then it just seems to disappear, as well as the validation recall at all thresholds never seemed to change at all. This was the same thing with precision as well, in fact it looked the exact same as the graphs below. For the recall at 0.5, we can see the training recall going up and down, but never really seems to flatten out throughout the entire training. The main issue is that the model did not improve or get worse from epochs 7-12, once it got to the 12th epoch is when all metrics began changing again. This brought us to believe that it may be due to a not deep enough model, needing more layers during training, and or not enough data, however we think that it was more so a not deep enough model simply because we can see all the metrics come back to life after the 12th epoch as well as most of the metrics, besides precision and recall, were actually changing throughout the whole training. 


 
7. Conclusions and Future Work 

 

Since the main challenge of the project is to create a model that can identify breast cancer, it is important to dive a bit deeper into the prediction results.  

As stated, the model predicted 0.49373 for the left breasts chance at having cancer, and 0.49372 for the right breast. If this were a real-life situation where this model was being used, this would be a little alarming for the doctors since there is a 0.49% chance whether the patient has cancer or not. Although this number is not extremely high, like 0.80+, one may not think of too much importance to do more testing and research on the patient.  

 

But this is where an issue may come into play, since models are not going to be 100% accurate all the time, when a higher number, like in this instance of 0.49% is shown for a patient's chance of cancer, it would not be a bad idea to investigate it more. If the number was significantly lower like 0.2 or even less, then it may not be important and can be guaranteed that the patient does not have cancer. However, when working with something as dangerous and life changing as cancer, any sort of sign or prediction of cancer being present should always be taken seriously.  

 

To conclude, the model turned out to be decent providing a 0.49% chance of the test patient having breast cancer. The model's metrics were also not horrible seeing an F1 score of 0.3314 and a binary accuracy score of 0.5018.  

  

For future recommendations, it would be easier if a more powerful computer were used to speed up the training time. Many hours were spent just trying to train the model to find the best parameters, yet this was incredibly difficult due to the model taking about 2-3 hours just to get to a point where it could be seen if things needed to be tweaked or not. If a better computer were provided, a higher number of epochs could have been used in training allowing more errors and faults to be caught, it also could allow a more in-depth model of more layers which could potentially fix some issues that arose.  

We could also use a GAN (Generative Adversarial Network) model which would create more synthetic images to train on, so we could create more cancerous train images. This would improve the model because we would have more images to train on that are more balanced between cancerous and noncancerous. Using a GAN would help not overfit the data because the data as it is currently very skewed towards noncancerous, but with a GAN we can even out the amount of data that is cancerous to noncancerous data.  

 
  

7. Appendix  

   

The link to our GitHub is https://github.com/fivefootbot/Mammography   

  

 

 
