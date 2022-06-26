# Pistachio Detective
I am learning Tensorflow so I thought an interesting application of my efforts would be to go online, find a data set, and do some basic image classification. 

### Kirmizi Pistachi: ![Kirmizi](.README_images/kirmizi_example.jpg)

### Siirt Pistachi: ![Siirt](.README_images/siirt_example.jpg)

I can't really tell the difference by eye, but the Neural Network did great!

## Technical Objectives
1. Make use of a Convolutional Neural Network. This is one of the requirements for the Tensorflow certification, but I'm generally interested in images and ML, so I thought it'd be worth practicing. 
2. Implementing a callback function to control epochs. I really don't want to overfit data, and callbacks seem to be a great way to do this. Along the way, I learned more about how `model.fit()` works.
3. Achieve reasonable accuracy. I honestly thought 90% would be more than enough, but with only 3 epochs, the CNN was able to reach 99%. That was a great experience for me because it encouraged me about the practical applications of Machine Learning.

### Callback Function at Work
![Callback](.README_images/callback.png)

## Data Set Used
I used the data set on the first link:
[https://www.muratkoklu.com/datasets/](Pistachio Data Set)
