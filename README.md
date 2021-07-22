# Eye of ML

Simple program to run yolov5 (large) object detection with video camera. 

Note this requires a stronger graphics card to run (tested on GTX1080Ti). 

It is recommended to **not** run `pip install -r requirements.txt` directly but to make them yourselves. Particularly, one is not sure how to get distributable from pytorch into requirements.txt hence requires you to install pytorch with cuda yourself, using the code defined from [Pytorch repository](https://pytorch.org/get-started/locally/). Particularly, choose the one with CUDA support according to your OS and CUDA version.  

Run the program with: `python -m render` or `python3 -m render` or perhaps `python -m render.py` or `python3 -m render.py` depending on which one works for your os and which one doesn't raise an error. 

### TensorFlow Models
If you would like to use TensorFlow models, there are additional installations required to run. 

If you are using Linux (or perhaps Mac OS?), you could run install.sh (after giving it permission using chmod). If it doesn't
execute properly, change `python` to `python3`. 

If you are on Windows, please follow the instruction [here](https://grpc.io/docs/protoc-installation/).
To summarize the instruction, go to [here](https://github.com/protocolbuffers/protobuf/releases) and download the 
`protoc-<version>-win64.zip`, unzip, set environment variable. More information refer to the instruction. Then, 
run the `install.bat` file. 

If you would like to use gpu, perhaps check with:
```python
import tensorflow as tf
tf.test.is_built_with_cuda()  # should output True
tf.config.list_physical_devices("GPU")  # Should found your GPU. 
```
If not, perhaps install `tensorflow-gpu>=2.5` after doing the above steps as the above step will 
install tensorflow. 

**Note for tensorflow**: Some large models that aren't download will require some time to download if this is the first
time open. The screen will freeze (i.e. **Not responding**) while downloading. Monitor the command prompt/terminal for more information on progress. 
Also for tensorflow models, it takes more time to load than for PyTorch's Yolov5 so requires wait for the model to load.
**Generally, it takes quite long to load with TF models. Exact timing is not available**. 