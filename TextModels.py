import tensorflow as tf
import TextPreprocessor

__textStringsList=[]
__textPredictionsDict={}

__distilBert=tf.saved_model.load('models/distilbert_classifier')

def __GetTextStringsRequest(request):
      if request.method =="POST":
        if 'textData' in request.form:
            # Recive text as one large string and split it to list of strings
            __textData=request.form['textData']
            __textStringsList=__textData.split(",")
            return __textStringsList

def FillPredictionsDict(request):
            __textStringsList=__GetTextStringsRequest(request)
            for text in __textStringsList:
                if(len(text)<=1):
                    continue
                # Preprocess the text and predict the toxicity
                text=TextPreprocessor.pipelineText(text)
                if(len(text)<=1) or text=="null" or (text in __textPredictionsDict):
                    continue
                print('==============================')
                print(text)
               # print(len(text))
                padding_mask,token_ids=TextPreprocessor.preprocess_text_list([text])
                print('==============================')

                predictions=__distilBert.signatures["serving_default"](padding_mask=padding_mask,
                                                                                token_ids=tf.constant(token_ids))
                # print(predictions)
                non_toxic,toxic=predictions['output_0'].numpy()[0]
                if non_toxic>toxic:
                    print("Non Toxic")
                    __textPredictionsDict[text]='Non Toxic'
                else:
                    print("Toxic")
                    __textPredictionsDict[text]='Toxic'
            return __textPredictionsDict