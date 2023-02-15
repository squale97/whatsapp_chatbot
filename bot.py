from flask import Flask, request, Response
import requests
#from googlesearch import search
from twilio.twiml.messaging_response import MessagingResponse
import json
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 

from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf 
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")
data = {}
with open('app/intents.json','r') as f:
    data = json.load(f)



# obtenir la racine des mots
lemmatizer = WordNetLemmatizer()

# Création des listes
words = []
classes = []
doc_X = []
doc_Y = []

# parcourir avec une boucle For toutes les intentions
# tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
# le tag associé à l'intention sont ajoutés aux listes correspondantes
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_Y.append(intent["tag"])
    
    # ajouter le tag aux classes s'il n'est pas déjà là 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatiser tous les mots du vocabulaire et les convertir en minuscule
# si les mots n'apparaissent pas dans la ponctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# trier le vocabulaire et les classes par ordre alphabétique et prendre le
# set pour s'assurer qu'il n'y a pas de doublons
words = sorted(set(words))
classes = sorted(set(classes))


#Preprocessing
training =[]
out_empty = [0] * len(classes)

# mise en place du modèle
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    output_row = list(out_empty)
    output_row[classes.index(doc_Y[idx])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
   # séparer les features et les labels target
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))
# définition de quelques paramètres
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200

#modèle Deep learning

model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))

adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

app = Flask(__name__)





def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens
def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)
def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list
def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
    else:
        result="Désolé je ne dispose pas d'infos la dessus"
  return result

"""
while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)
"""

@app.route('/bot', methods=['POST'])
def bot():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    intents = pred_class(incoming_msg, words, classes)
    result = get_response(intents, data)
    msg.body(result)
    responded = True
    """
    if 'quote' in incoming_msg:
        # return a quote
        r = requests.get('https://api.quotable.io/random')
        if r.status_code == 200:
            data = r.json()
            quote = f'{data["content"]} ({data["author"]})'
        else:
            quote = 'I could not retrieve a quote at this time, sorry.'
        msg.body(quote)
        responded = True
    if 'cat' in incoming_msg:
        # return a cat pic
        msg.media('https://cataas.com/cat')
        responded = True
    if 'casier' in incoming_msg:
        msg.body("Pour le casier judicaire il faut une demande timbrée, une somme de 300frs et un extrait de naissance")
        responded = True
    if not responded:
        msg.body('I only know about famous quotes and cats, sorry!')
    """
    print(str(resp))
    return Response(str(resp), mimetype="application/xml")

if __name__ == '__main__':
    app.run()

