import tensorflow as tf
import TextPreprocessor

textStringsList=[]
textPredictionsDict={}

distilBert=tf.saved_model.load('models/distilbert_classifier')

def GetTextStringsRequest(request):
      if request.method =="POST":
        if 'textData' in request.form:
            # Recive text as one large string and split it to list of strings
            textData=request.form['textData']
            textStringsList=textData.split(",")
            return textStringsList

def FillPredictionsDict(request):
            textStringsList=GetTextStringsRequest(request)
            for text in textStringsList:
                if(len(text)<=1):
                    continue
                # Preprocess the text and predict the toxicity
                text=TextPreprocessor.pipelineText(text)
                if(len(text)<=1) or text=="null" or (text in textPredictionsDict):
                    continue
                print('==============================')
                print(text)
               # print(len(text))
                padding_mask,token_ids=TextPreprocessor.preprocess_text_list([text])
                print('==============================')

                predictions=distilBert.signatures["serving_default"](padding_mask=padding_mask,
                                                                                token_ids=tf.constant(token_ids))
                # print(predictions)
                non_toxic,toxic=predictions['output_0'].numpy()[0]
                if non_toxic>toxic:
                    print("Non Toxic")
                    textPredictionsDict[text]='Non Toxic'
                else:
                    print("Toxic")
                    textPredictionsDict[text]='Toxic'
            return textPredictionsDict