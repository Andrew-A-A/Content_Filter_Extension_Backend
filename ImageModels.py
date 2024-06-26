import tensorflow as tf
import requests
import ImagePreprocessor 
from PIL import Image
import io

mobileNetV3 = tf.saved_model.load("models/MobileNetV3")

efficientNet = tf.saved_model.load("models/EfficientNet")

__imagesList=[]
binaryPredictions={}
multiClassPredictions={}

def __GetImageRequest(request):
    # Check if method recived is correct
    if request.method =="POST":
        # Check if recived request contains images list
        if "images" in request.form:
            # Get images urls as one string
            images=request.form['images']
            # Split the string to get the list of images
            __imagesList=images.split(',')
            return __imagesList

def FillBinaryDict(request):
            __imagesList=__GetImageRequest(request)
            for image in __imagesList:
                if image=="":
                    continue
                # Load the image from the url
                response = requests.get(image)
                predict_image=ImagePreprocessor.ImgPreprocess(response)
                # Get the prediction from the MobileNetV3 model
                binaryPrediction=mobileNetV3.signatures["serving_default"](predict_image)
                # Map the prediction to the correct class (Binary)
                non_violence,violence=binaryPrediction['dense'].numpy()[0]
                if non_violence<violence:
                    prediction="Violence"
                else:
                    prediction="Non-Violence"
                binaryPredictions[image]=prediction
            return binaryPredictions

def FillMultiClassDict(request):
            __imagesList=__GetImageRequest(request)
            for image in __imagesList:
                if image=="":
                    continue
                # Load the image from the url
                response = requests.get(image)
                predict_image=ImagePreprocessor.ImgPreprocess(response,True)
                multiClassPrediction = efficientNet.signatures["serving_default"](predict_image)
                # Map the prediction to the correct class
                accident, damaged_buildings, fire, normal =multiClassPrediction['output_0'].numpy()[0]
                predictions_dict={"fire":fire,"accident":accident,'normal':normal,"damaged_buildings":damaged_buildings}
                multiClassPredictions[image]=max(predictions_dict,key=predictions_dict.get)
            return multiClassPredictions

   