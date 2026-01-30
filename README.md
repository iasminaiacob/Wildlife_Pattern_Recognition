# Wildlife Pattern Recognition using Camera Trap Images
This project builds an automated system for identifying animal species in camera trap images and analyzing wildlife occurrence patterns using deep learning features and classical machine learning.

Using a pretrained ResNet18 convolutional neural network as a feature extractor, images are converted into fixed-length embeddings that are then classified with logistic regression. The project also explores spatial patterns of species occurrence and evaluates scalability using parallel processing with Dask.

---

## Project Goals

- Automatically classify animal species from camera trap images  
- Analyze occurrence patterns by species and camera location  
- Explore scalable processing for large image datasets  

---

## Dataset

ECCV 2018 Camera Trap Dataset  
Images and COCO-style JSON annotations containing image metadata, species labels, and locations.

After filtering and subsampling:
- 11,075 images  
- 10 species  

Top species include bobcat, cat, opossum, coyote, raccoon, dog, squirrel, rabbit, skunk, and bird.

---

## Method Overview

1. Extract image and annotation archives  
2. Build metadata table linking images to species and locations  
3. Generate 512-dimensional image embeddings using pretrained ResNet18  
4. Train logistic regression classifier on embeddings  
5. Evaluate classification performance  
6. Analyze spatial occurrence patterns  
7. Visualize results and errors  

---

## Model

- Backbone: ResNet18 (ImageNet pretrained)  
- Feature dimension: 512  
- Classifier: Logistic Regression  
- Train/Test split: 80/20 (stratified)

---

## Results

- Overall accuracy: 75%  
- Macro F1-score: 0.77  

Best performing species: dog, squirrel  
More challenging species: bobcat, opossum  

Confusion matrices, per-class accuracy charts, and failure examples are provided in the `outputs/` directory.

---

## Pattern Analysis

- Species frequency distributions  
- Species-by-location heatmaps  
- Visualization of embedding space (t-SNE)  
- Qualitative inspection of misclassifications  

Temporal analysis was limited due to missing timestamps in a subset of data.

---

## Scalability

Dask is used to parallelize image embedding extraction and demonstrate distributed tabular operations.  
For small datasets, Pandas is faster, but Dask becomes useful when computation or memory exceeds a single machine.
