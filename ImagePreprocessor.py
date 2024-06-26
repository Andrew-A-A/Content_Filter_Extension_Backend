from PIL import Image
from io import BytesIO
import io
import tensorflow as tf



def ImgPreprocess(img,isMulti=False):
     preprocssedImage = Image.open(io.BytesIO(img.content)).resize((224,224)).convert('RGB')
     preprocssedImage = tf.keras.preprocessing.image.img_to_array(preprocssedImage)
     preprocssedImage = tf.expand_dims(preprocssedImage, axis=0)
     if isMulti:
            preprocssedImage = tf.keras.applications.efficientnet_v2.preprocess_input(preprocssedImage)
     return preprocssedImage
