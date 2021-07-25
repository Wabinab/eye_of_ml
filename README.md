# Eye of ML

Simple program to run yolov5 (large) object detection with video camera. 

Note this requires a stronger graphics card to run (tested on GTX1080Ti). 

It is recommended to **not** run `pip install -r requirements.txt` directly but to make them yourselves. Particularly, one is not sure how to get distributable from pytorch into requirements.txt hence requires you to install pytorch with cuda yourself, using the code defined from [Pytorch repository](https://pytorch.org/get-started/locally/). Particularly, choose the one with CUDA support according to your OS and CUDA version.  

Run the program with: `python -m render` or `python3 -m render` or perhaps `python -m render.py` or `python3 -m render.py` depending on which one works for your os and which one doesn't raise an error. 

Credits to https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/tree/master for the EfficientDet implementations. 
Some of the folders and utilities (python files) in this directory is a direct fork from the files
so the model could be loaded and use. 

Thanks to https://ultralytics.com/ for their color palette usage in this project.

To use EfficientDet, it is required to first run `efficientdet.bat` for Windows. 
Linux support may come later (or change `.bat` to `.sh` and it will run since it's the same code anyways). 