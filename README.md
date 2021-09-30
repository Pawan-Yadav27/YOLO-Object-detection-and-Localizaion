<p align="center">
<h3 align="center">YOLO: Object-detection-and-Localization</h3>
<div align="center">
<p> "You Only Look Once" (YOLO) is a popular algorithm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes. </p>


</div>

------------------------------------------
### Welcome to YOLO : 

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.

![YOLO_v2 COCO model with test_yolo defaults](person.jpg)

</div>
--------------------------------------------------------------------------------

## Requirements

- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/) (For Keras model serialization.)
- [Pillow](https://pillow.readthedocs.io/) (For rendering test results.)
- [Python 3](https://www.python.org/)
- [pydot-ng](https://github.com/pydot/pydot-ng) (Optional for plotting model.)

-----------------------------------------------------------------------------------------
### Inputs & Outputs :
- The **input** is batch of images, each image has the shape (m , 416 , 416 , 3) 
- The **output** is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers (p_c, b_x, b_y, b_h, b_w, C) . Here C is an 80-dimensional vector, so each bounding box is represented by 85 numbers. 

### Anchor Boxes : 
- Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that represent the different classes.  For this projext, 3 anchor boxes were chosen. 
- The dimension for anchor boxes is the second to last dimension in the encoding: (m, n_H,n_W,anchors,classes).

### Model Architecutre :
- The YOLO architecture is: IMAGE (m, 416, 416, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 3, 85). 
- The YOLO architecture is picturized in 'yolo.png' file.


### Quick Start

(1) Download Darknet model cfg and weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).


(2) Convert the Darknet YOLO_v2 model to a Keras model, or can use converted keras model in root directory.


(3) Execute the 'train-model.py' to download the trained model. 


(4) Class score

   (A) Now, for each box (of each cell) we will compute the following element-wise product and extract a probability that the box contains a certain class.  
   (B) The class score is score_{c,i} = p_c * c_i: the probability that there is an object p_c times the probability that the object is a certain class c_i.
     
     
(5) Visualizing classes

   (A) Here's one way to visualize what YOLO is predicting on an image:
   (B) For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across the 80 classes, one maximum for each of the 3 anchor boxes).
   (C) Color that grid cell according to what object that grid cell considers the most likely.
   (D) **Note that this visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm.**
    
    
(6) Filtering with a threshold on class scores

  (A) You are going to first apply a filter by thresholding. You would like to get rid of any box for which the class "score" is less than a chosen threshold. 
  (B) The model gives a total of 19x19x3x85 numbers, with each box described by 85 numbers. It is convenient to rearrange the (19,19,3,85) (or (19,19,255)) dimensional tensor into the following variables:  
        (i) `box_confidence`: tensor of shape (19 * 19, 3, 1) containing p_c (confidence probability that there's some object) for each of the 3 boxes predicted in each of the 19x19 cells.
        (ii) `boxes`: tensor of shape (19 * 19, 3, 4) containing the midpoint and dimensions (b_x, b_y, b_h, b_w) for each of the 3 boxes in each cell.
        (iii) `box_class_probs`: tensor of shape (19 * 19, 3, 80) containing the "class probabilities" (c_1, c_2, ... c_{80}) for each of the 80 classes for each of the 3 boxes per cell.
        
        
(7) Non-Max suppression

   (A) Now we have boxes for which the model had assigned a high probability, but this is still too many boxes. we'd like to reduce the algorithm's output to a much smaller number of detected objects.  
   (B) To do so, we'll use **non-max suppression**. Specifically, we'll use **"Intersection over Union(IoU)"**. If you are not familier with IoU, google it.

</div>

---------------------------------------------------------------------------------------------------------------------------------------

### Summary for YOLO:
- Input image (416, 416, 3)
- The input image goes through a CNN, resulting in a (19,19,3,85) dimensional output. 
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 255):
    - Each cell in a 19x19 grid over the input image gives 255 numbers. 
    - 255 = 3 x 85 because each cell contains predictions for 3 boxes, corresponding to 3 anchor boxes. 
    - 85 = 5 + 80 where 5 is because (p_c, b_x, b_y, b_h, b_w) has 5 numbers, and 80 is the number of classes we'd like to detect
- You then select only few boxes based on:
    - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
- This gives you YOLO's final output. 

</div> 

------------------------------------------------------------------------------

### Results- 
    
![image](https://user-images.githubusercontent.com/73088379/132945948-ac9a2b5e-347d-436d-8ffe-9807c71c4f3e.png)


--------------------------------------------------------------------------------

