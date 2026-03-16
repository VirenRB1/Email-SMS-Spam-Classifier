import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# This must exist before unpickling transform.pkl, because the pickle refers to
# __main__.transform_text when it was created.


def transform_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load the model and vectorizer
transform_text = pickle.load(open("transform.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


def predict(text):
    transformed_text = transform_text(text)
    vectorized_text = tfidf.transform([transformed_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    return prediction


input_text = input("Enter a message to classify as spam or not spam: ")

result = predict(input_text)
if result == 1:
    print("Spam.")
else:
    print("Not Spam.")
