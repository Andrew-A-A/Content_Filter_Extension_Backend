from flask import Flask, render_template, request , jsonify
import os
import ImageModels
import TextModels

app = Flask(__name__)
if __name__ == "__main__":
    app.run(debug=True)

# Method used to get the list of images from extension
@app.route('/upload-urls',methods=["POST"])
def getImagesList():
    # Check if method recived is correct
    if request.method!="POST":
        return jsonify({"BackendError": "Error in request"})
    elif "images" not in request.form:
        return jsonify({"BackendError":"Images not sent"})
    
    binary_predictions=ImageModels.FillBinaryDict(request)
    multi_class_predictions=ImageModels.FillMultiClassDict(request)
    # Return the both multi-class and binary predictions to the frontend
    return jsonify({'msg': 'success', 'prediction': binary_predictions,'mulit-class-prediction':multi_class_predictions})

# Methon used to get list of strings from extension
@app.route('/upload-text',methods=['POST'])
def getStringsList():
    if request.method!="POST":
        return jsonify({"BackendError": "Error in request"})
    elif "textData" not in request.form:
        return jsonify({"BackendError":"Text not sent"})
    text_prediction_dict=TextModels.FillPredictionsDict(request)
    return jsonify( {"TextPrediction":text_prediction_dict})



@app.route("/")
def home():
    return render_template("helloWorld.html")

# Used when image is uploaded from the helloWorld.html
@app.route("/upload-image", methods=["","POST"])
def upload_image():
    if request.method == "POST":
        if "image" in request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            filename = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            predict_image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
            predict_image = tf.keras.preprocessing.image.img_to_array(predict_image)
            predict_image = tf.expand_dims(predict_image, axis=0)
            prediction=mobileNet_image_model.signatures["serving_default"](predict_image)
            # input_tensor = tf.convert_to_tensor(eff_input, dtype=tf.float32)
            # Extracting the numpy array from the tensor
            non_violence,violence=prediction['dense'].numpy()[0]
            if non_violence<violence:
                prediction="Violence"
            else:
                prediction="Non-Violence"
            return render_template("upload_image.html", uploaded_image=filename, model_prediction=prediction )
    return render_template("upload_image.html")

# Used when image is uploaded from the upload_image.html
@app.route('/process-image', methods=["POST"])
def process_image():
 if request.method == "POST":
    if "image" in request.files:
        image = request.files["image"]
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
        filename = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
        predict_image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
        predict_image = tf.keras.preprocessing.image.img_to_array(predict_image)
        predict_image = tf.expand_dims(predict_image, axis=0)
        prediction=mobileNet_image_model.predict(predict_image)
        if prediction[0][0]<prediction[0][1]:
            prediction="Violence"
        else:
            prediction="Non-Violence"

        return jsonify({'msg': 'success', 'prediction': [prediction]})
    return jsonify({'Error': 'BackendError'})

# Dummy route to test only requests
@app.route('/test', methods=["POST"])
def test():
    if request.method == "POST":
         if "image" in request.files:
             return jsonify({'msg': 'Bravo'})
         return jsonify({'msg': 'Bad'})
    
# Used in testing phase to upload the images to server then send it to the mobileNet model
app.config["IMAGE_UPLOADS"] = "static/Uploads/"
#endregion
