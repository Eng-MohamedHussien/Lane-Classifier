# Lane-Classifier

In order to get rid of noise in our lane detection project when there is no lanes in the image . 
We decided to create classifier to classify images with lanes or not.
Incase of image has no lane input image will be the output image with no change.
Otherwise output from stage before fully connected layer in our classifier will be input to our decoder part in our lane detection architecture to segment lane.
