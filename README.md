# Eye of ML

Simple program to run yolov5 (large) object detection with video camera. 

Note this requires a stronger graphics card to run (tested on GTX1080Ti). 

It is recommended to **not** run `pip install -r requirements.txt` directly but to make them yourselves. Particularly, one is not sure how to get distributable from pytorch into requirements.txt hence requires you to install pytorch with cuda yourself, using the code defined from [Pytorch repository](https://pytorch.org/get-started/locally/). Particularly, choose the one with CUDA support according to your OS and CUDA version.  
