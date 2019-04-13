# Lane-Classifier

In order to get rid of noise in our lane detection project when there is no lanes in the image . 
We decided to create classifier to classify images with lanes or not.
Incase of image has no lane input image will be the output image with no change.
Otherwise output from stage before fully connected layer in our classifier will be input to our decoder part in our lane detection architecture to segment lane.

## Criteria :

1- the car must be in center and has right and left line.

2- if one line appears and the other one isn't so it isnot considered as lane.

3- if one line appears and the other one appears but partially and pale it isnot considered also as lane.

4- if there is space large enough between car and lane it isnot considered also as lane.


Dataset added by <a href="https://github.com/teamleader6">Mohamed Hussien</a> show files from  <a href="">here</a>
