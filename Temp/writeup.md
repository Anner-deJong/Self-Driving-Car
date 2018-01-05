# **CarND Project 3 Writeup** 
# **Behavioral Cloning** 


### This writeup includes a description for each of the following goals/steps:

#### [1. Data Acquisition & Description,](#Data-Acquisition-&-Description)
#### [2. Model Architecture,](#Model-Architecture)
#### [3. Training Details, and](#Training-Details)
#### [4. Autonomous Mode Simulator Performance](#Autonomous-Mode-Simulator-Performance)

##### NB This repository does NOT contain either the data or the augmented data. Download/create these yourself and put them in the correct directory.

[//]: # (Image References)

[FLIP]: ./Images/FLIP.jpg "Flipping augmentation example"
[LCR]: ./Images/LCR.jpg "Left center and right view example"
[Crop]: ./Images/CROP.jpg "Cropping example"
[CURVE_1]: ./Images/training_curve_first_7(6)_Epochs.jpg "Training curve for first training cycle"
[CURVE_2]: ./Images/training_curve_second_5_Epochs.jpg "Training curve for second training cycle"
[VIDEO]: ./run1_second_train.mp4 "Recorded autonomous lap"

---

### Data Acquisition & Description
[//]: # ([(page top)](#CarND-Project-3-Writeup)

Data is acquired via the straightforward method included in Udacity's simulator.
Data augmentation is applied through flipping each image and multiplying the corresponding steering angle by -1.

![alt text][FLIP]

Furthermore, the left and right side camera images are also included, by adding or subtracting a steering angle correction factor (see [Training Details](#Training-Details) below)

![alt text][LCR]

The captured images are from a single lap on track 2 (for generalization purposes) and 4 or 5 laps on track 1. The laps on track 1 include 2 reverse laps as well for generalization (the left steering bias of track 1 is already broken by flipping each image).
Furthermore several recovery patterns are recorded as training images as well through repeating this pipeline:

1. Steering to the side of the road
2. Facing the edge
3. Start recording
4. Turn sharlpy away from the edge
5. Easen the steering angle and reversing it so the car ends up straight in the middle of the road
6. Stop recording

### Model Architecture

In the lectures it was suggested that [NVIDIA's model](https://arxiv.org/abs/1604.07316) might be a good solution architecture. Personally I believe however this should be overkill, considering the differences in application: a real world case versus a limited simulation environment.

Therefore I have build a simpler model in keras (perhaps a sort of skeletal version of NVIDIA's architecture if you will). However, a fellow Udacity student made me realize that since the NVIDIA model neatly alines their last conv layer to end with a shape of 1x18x64, their fully connected layer afterwards does not require by far as much neurons as my current version.

    # In-model-preprocessing
    model.add(Cropping2D(cropping=((TOP_CROP, BOT_CROP), (0, 0)), input_shape=(H, W, C)))
    model.add(AveragePooling2D())
    model.add(Lambda(lambda x: x / 255 - 0.5)) # normalize
    # Conv1
    model.add(Convolution2D(32, 3, 3, init='glorot_uniform'))
    # Conv2
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform'))
    model.add(Dropout(p=0.8))
    # Conv3
    model.add(Convolution2D(128, 3, 3, init='glorot_uniform'))
    model.add(Dropout(p=0.7))
    # Conv4
    model.add(Convolution2D(256, 3, 3, init='glorot_uniform'))
    model.add(Dropout(p=0.6))
    # FC5
    model.add(Flatten())
    model.add(Dense(256, init='glorot_uniform'))
    model.add(Dropout(p=0.5))
    # FC6
    model.add(Dense(128, init='glorot_uniform'))
    model.add(Dropout(p=0.5))
    # FC7
    model.add(Dense(1, init='glorot_uniform'))


#### In-model preprocessing
First some in-model preprocessing is applied, in order for the same preprocessing to be applied to input in inference mode.
In line with the lectures, the top part and the bottom part of the input images dont contain valuable information, and are therefore cropped out (see Hyperparameters in the next section for the detailed crop sizes):

![alt text][CROP]

In line with [Paul Heraty](https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb) I believe the input resolution is higher than necessary for infering steering angles. Secondly therefore in the preproccesing pipeline is a downsampling of the resolution in both dimensions by 2 (implemented as a Keras AveragePool function).

Thirdly, the preprocessing includes the regular normalizing and 0 centering of the pixel values.

#### Trainable layers
Following the preprocessing is the actual trainable model, consisting of 7 layers in total.

Each of the 7 layers includes Batch Norm and uses with Xavier initializations, and all but the last layer use ReLU activations. The first three conv layers include a MaxPool of size 2x2 (the last conv layer ends up with HxW = 1x16 so no further Pooling is applied). Dropout is applied to the 2nd until 6th layer in an increasingly aggressive regularization manner (increasing dropout rates (0.8, 0.7, 0.6, 0.5, 0.5)).

#### More NVIDIA alike model

Since the amount of weights blew up in the above mentioned model, I build another model more similar to NVIDIA's model, in that it has the same number of layers, and in that the last conv layer ends up with one dimension of size 1. Each layer except the last one again uses batch norm and ReLU activations.

    # Preprocessing
    model = Sequential()
    model.add(AveragePooling2D(input_shape=(H, W, C)))
    model.add(Lambda(lambda x: x / 255 - 0.5)) # normalize
    # Conv1
    model.add(Convolution2D(24, 5, 5, init='glorot_uniform', border_mode='same', subsample=(2,2)))
    # Conv2
    model.add(Convolution2D(36, 5, 5, init='glorot_uniform', subsample=(2,2)))
    model.add(Dropout(p=0.9))
    # Conv3
    model.add(Convolution2D(48, 5, 5, init='glorot_uniform', subsample=(2, 2), border_mode='same'))
    model.add(Dropout(p=0.9))
    # Conv4
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform'))
    model.add(Dropout(p=0.8))
    # Conv5
    model.add(Convolution2D(64, 3, 3, init='glorot_uniform'))
    model.add(Dropout(p=0.8))
    # Flatten
    model.add(Flatten())
    # FC6
    model.add(Dense(100, init='glorot_uniform'))
    model.add(Dropout(p=0.7))
    # FC8
    model.add(Dense(50, init='glorot_uniform'))
    model.add(Dropout(p=0.7))
    # FC9
    model.add(Dense(10, init='glorot_uniform'))
    # FC10
    model.add(Dense(1, init='glorot_uniform'))

In the hope this model performs better also on track 2, I have left out the cropping, as the mountainous winding roads going up and down likely contain more information in the outer regions of the images.

### Training Details

#### Initial model
The training is done in training cycles, making use of the model save and load functionality in Keras. This eases the computation burden, and allows for simulator testing in between epochs. The model.py and model.ipynb in this repository exhibit the code used at the last training cycle. Training was performed with the Adam optimizer provided by Keras, and if not mentioned differently per cycle, these hyperparameters were used:

    # Hyperparameters
    EPOCHS     = 5
    BATCH_SIZE = 256
    LRN_RATE   = 0.001
    STEER_COR  = 0.2    # correction to be added/subtracted to left/right images respectively
    TOP_CROP   = 50     # number of pixels to be cropped from top side
    BOT_CROP   = 22     # number of pixels to be cropped from bottom side

##### First training cycle

    EPOCHS     = 7

![alt text][CURVE_1]

(For a better view the above graph skips the values for the first epoch.) This looks like a healthy training curve, if it was not for the constant lower validation loss compared to training loss. I've rechecked this behavior a couple of times, yet this is what Keras keeps outputting. The training and validation sets are properly shuffled before they are separated, so my only and best lucky guess is that it is somehow caused by the MSE loss incorporating the different training and validation set sizes. Although this behaviour is very counter-ML-intuitive, both of the losses are at least still decreasing (and actually the AI is not able to drive along the track), so I tried another training cycle.

##### Second training cycle

    BATCH_SIZE = 192
    LRN_RATE   = 0.0005

![alt text][CURVE_2]

The losses keep decreasing, which is a very good sign, although they decrease quite slowly. However, after the second training cycle, the AI is actually able to drive a full lap!

However, the car *does* drive over the line markings on the right side of the road twice (yet still without touching the hightened bumpers on the side of the road). For this reason I tried training more.

Also, I realized I was using BGR format during training, but for inference RGB is used. (Apparently that doesnt matter so much, which is not that weird if you think about it. This means we might also be able to convert to a grey scale input, or even a canny edge input. I did not explore these options). 

##### Third training cycle

    EPOCHS     = 3

For this training cycle I changed to a proper RGB input, and also aqcuired a new dataset with more recovery patterns (as descibed in [Data Acquisition & Description](#Data-Acquisition-&-Description) above). A graph with this new dataset would give a false comparison with the graphs above so it is not included.

I used a stronger learning rate again (the original one) since I expect the model had to shift weights a bit more with the changed input. Furthermore, I wanted to decrease the amount of dropout applied in this later training cycle with

    model.get_layer('name').p = x
    
I was able to change the dropout rates, but I am not completely familiar with the implementation. Changing dropout rates requires a change in the dropout correction factor as well (that should be automatically applied by Keras when initializing a dropout layer). Not sure whether my correction factor was changing properly with the changed rates, and seeing an increase in loss, I did not change the dropout rates in the end and did not explore this optimization further.

Testing after this third training cycle broke the AI, it was not able to complete a lap on track 1 anymore. As such, I focussed more attention to developing the NVIDIA alike model.

#### NVIDIA Alike Model

    # Hyperparameters
    EPOCHS     = 35
    BATCH_SIZE = 256
    LRN_RATE   = 0.002
    LR_DECAY   = 0.07
    STEER_COR  = 0.2

In a sort of all-in maneuver I acquired more data on track 2 as well, and tried to train this model with a much larger number of epochs. Because of this larger number, I also included a learning rate decay, and therefore started the initial learning rate a bit higher.

Stressed for time and resources, I did not test this model yet (it is still training while I am writing this).
I hope to update this repository after the training on this model is finished.



### Autonomous Mode Simulator Performance

Here is the ourput of the 
The final result can be seen in this [video](/view/Autonomous_mode_second_train_cycle.mp4) 
<video width="320" height="240" controls>
  <source src="./run1_second_train.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

![alt text][VIDEO]

