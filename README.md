
<center>

![exp](/assets/exp.jfif) 

# Image Colorization

</center>

Pytorch implementation of VGG architecture with 3 different loss functions: L2 norm loss, balanced class cross-entropy loss 
([Balanced class cross-entropy loss reference Paper](https://arxiv.org/pdf/1603.08511.pdf)) and custom defined focal loss using the precedent defined loss, implemented end to end.

We present deployed versions of 2 variants: L2 norm loss and focal loss with L2 norm.
The models were trained in google cloud, using VM instances of specificaions: n1-highmem-16(16 vCPUs, 104 GB memory), GPU: 4 x NVIDIA Tesla V100

The training was parallelized along 3 machines, training time taking about 1h~2h of average each.

### Reference Archictecture

The balanced class cross-entropy architecture is showed bellow, according to [Balanced class cross-entropy loss reference Paper](https://arxiv.org/pdf/1603.08511.pdf)

![architecture](/assets/ach.jpeg)
![architecture2](/assets/arch.jpeg)

We've used in the project a slightly different version removing the penultimate conv layer (in blue in the figure). The original paper used this conv layer of output 313 activation filters 
to create a color probability distribution for each pixel, in a quantized color space of dimension 313. The color ab frame in the original paper is then created using a decode function.

In our implementation, the color ab frame output is obtained of the output of a conv layer of 2-out activations filters, immediatly after the conv8 layer, as showed bellow:

![architecture3](/assets/arch2.jpeg)


### Link to Presentation
 
[Presentation](https://docs.google.com/presentation/d/1bFiRyjH0R1xFo_R_IJFOd3BhUl6sZHinj6vV4ZywLqk/edit#slide=id.p)



### Link to Datasets

- [Sun Images Objects](http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz) : Scene benchmark (397 scene categories), tar file (37GB)
- [Sun Images Scenes](http://groups.csail.mit.edu/vision/SUN1old/SUN397.tar) : 16,873 images, tar file (7.3GB)


### Download trained models:


- [L2 loss Model](https://storage.googleapis.com/left-shift/model_l2.pt) 
- [Focal loss Model](https://storage.googleapis.com/left-shift/model_l2_focal.pt) 

### Link to results drive folder

For each model we tested our model against Sun Images Objects Dataset(training dataset, 16,873 images) and Sun Images Scenes (37GB) and selected the best and worst results,
based in the model loss after training.

These images are splitted in 4 folders in the drive, which one containing 2 subfloder: best and worst. The predictions images and the real images can be distinguish according to pred 
and real anotations in the file names

[Results](https://drive.google.com/drive/folders/1mPM673EesECNAtnXNATPIfve0hhQnuSz?usp=sharing) 



### Reference papers and useful links 

- [Colorful Image Colorization, ECCV 2016](http://richzhang.github.io/colorization/)
- [Colorful Image Colorization Paper](https://arxiv.org/pdf/1603.08511.pdf)
- [DeOldify projects](https://github.com/jantic/DeOldify)




### Dependencies

- scikit-image
- pytorch
- matplotlib 
- numpy 


### Folder Structure

![folder structure](/assets/structure.jpeg)


