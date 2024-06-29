from tkinter import *
from PIL import ImageTk, Image
import tkinter.messagebox as tkMessageBox
from tkinter import filedialog
import ctypes, random
import tkinter.scrolledtext as scrolledtext
import easyocr, os, re, openai
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image
import pytesseract
import io
import os,threading

def server():
    os.system('python manage.py runserver')
threading.Thread(target=server).start()
# Load the trained model
model = load_model('model.h5')
df_train = pd.read_csv('data2.csv')

le = LabelEncoder()
le.fit(df_train['Substance'])

home = Tk()
home.title("ANALYSIS OF FOOD ADDITIVES IN PACKAGED FOODS")
directory = "./"
img = Image.open(directory + "/assets/home.jpeg")
img = ImageTk.PhotoImage(img)
panel = Label(home, image=img)
panel.pack(side="top", fill="both", expand="yes")
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
lt = [w, h]
a = str(lt[0] // 2 - 600)
b = str(lt[1] // 2 - 360)
home.geometry("1200x720+" + a + "+" + b)
home.resizable(0, 0)
file = ''
reader = easyocr.Reader(['en'], gpu=True)


def google_vision_ocr(image_path):
    # Set up Google Cloud Vision API client
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Perform OCR using Google Cloud Vision API
    response = client.text_detection(image=image)
    texts = response.text_annotations

    google_vision_text = ""
    for text in texts:
        google_vision_text += text.description + ' '

    return google_vision_text.strip()


def pytesseract_ocr(image_path):
    # Perform OCR using pytesseract
    image = Image.open(image_path)
    pytesseract_text = pytesseract.image_to_string(image)

    return pytesseract_text.strip()


def combined_ocr(image_path):
    # Get OCR results from both Google Cloud Vision API and pytesseract
    google_vision_text = google_vision_ocr(image_path)
    pytesseract_text = pytesseract_ocr(image_path)

    return {
        "google_vision_text": google_vision_text,
        "pytesseract_text": pytesseract_text
    }


def view(inp):
    import tkinter as tk
    from tkinter import ttk

    data = []
    for i in inp.split('\n'):
        data.append(i.split(' --- '))

    root = tk.Tk()
    root.title("Substance Impact Treeview")

    # Create Treeview
    tree = ttk.Treeview(root, columns=("substance", "effect", "impact", "risk_factors"), show="headings")

    # Add headings
    tree.heading("substance", text="Substance")
    tree.heading("effect", text="Effect")
    tree.heading("impact", text="Impact")
    tree.heading("risk_factors", text="Risk Factor(s)")

    # Add data to the tree
    for item in data:
        try:
            tree.insert("", "end", values=item)
        except:
            pass

    # Styling
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))
    style.configure("Treeview", font=("Helvetica", 10))

    # Scrollbars
    yscroll = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    xscroll = ttk.Scrollbar(root, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

    # Pack everything
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")

    # Adjust column weights so they expand proportionally
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    root.mainloop()


def preprocess_input(user_input, le):
    try:
        # Encode the user input using the loaded label encoder
        user_input_encoded = le.transform([user_input])
        return user_input_encoded
    except ValueError:
        # Handle unseen input
        print(f"Warning: Unseen input '{user_input}' encountered. You may want to retrain the model with this input.")
        return None


def test_model(user_input, model, le):
    # Preprocess user input
    user_input_encoded = preprocess_input(user_input.upper().strip(), le)

    try:
        effect = df_train.loc[df_train['Substance'] == user_input.upper().strip()]['Effect'].tolist()
        effect = random.choice(effect)
    except:
        effect = 'None'
    try:
        risk_factors = df_train.loc[df_train['Substance'] == user_input.upper().strip()]['Risk Factor(s)'].tolist()
        risk_factors = random.choice(risk_factors)
    except:
        risk_factors = 'None'

    if user_input_encoded is not None:
        # Make predictions
        prediction = model.predict(user_input_encoded)

        # Interpret the prediction
        result = "Dangerous" if prediction > 0.5 else "Normal"

        return f"{effect} --- {result} --- {risk_factors}"
    else:
        return f"{effect} --- Unknown --- {risk_factors}"


def Exit():
    import webbrowser
    webbrowser.open("http://localhost:8000")


def fileopen():
    global file
    file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image",
                                      filetypes=(("images", ".png"), ("images", ".jpg"), ("images", ".jpeg")))


def process():
    if file != '' and file is not None:
        # Read text from an image
        result = reader.readtext(file)
        out = ''
        # Display the result
        for detection in result:
            out += '\n' + detection[1]  # The detected text

        # Extract capitalized words (potential food additives)
        additives = re.findall(r'\b[A-Z][a-z]*\b', out)

        # Join the additives into a comma-separated string
        additives_string = ', '.join(additives)

        txt.delete("1.0", "end")
        txt.insert("1.0", additives_string)


def analyse():
    out = ''
    for i in txt.get("1.0", 'end-1c').split(','):
        out += i + ' --- ' + test_model(i, model, le) + '\n'

    txt.delete("1.0", "end")
    txt.insert("1.0", out)
    view(out)


def about():
    tkMessageBox.showinfo(
        'About Us', """Food is an essential element for all living organisms. Processed foods have a very 
limited life, in order to preserve them for a longer time and to have good flavor, texture, 
and taste, food additives and preservatives are used. This study is taken to investigate the 
effects of food additives and preservatives on human health and did research on different 
Optical Character Recognition (OCR) algorithms. In this project, we propose an OCR-
based algorithm to identify food additives and preservatives in packaged foods by 
analyzing the contents of the package and finding their effect on human health. Here, the 
image of the contents of the package is applied to Image enhancement techniques such as 
to increase the accuracy of the OCR. The Google Cloud Vision API detects text from 
enhanced images based on CNN with high accuracy. The extracted text is filtered to remove 
undesirable words using RegEx. The filtered keywords are used to search in the predefined 
dataset where data is taken from the Standard Food Safety Organizations such as FDA and 
FSSAI. By using Google Cloud Vision API, we can extract text with an accuracy of 99.1%.""")


photo = Image.open(directory + "assets/1.jpeg")
img3 = ImageTk.PhotoImage(photo)
b2 = Button(home, highlightthickness=0, bd=0, activebackground="#e4e4e4", image=img3, command=fileopen)
b2.place(x=69, y=221)

photo = Image.open(directory + "assets/2.jpeg")
img2 = ImageTk.PhotoImage(photo)
b1 = Button(home, highlightthickness=0, bd=0, activebackground="white", image=img2, command=process)
b1.place(x=69, y=330)

photo = Image.open(directory + "assets/3.jpeg")
img4 = ImageTk.PhotoImage(photo)
b1 = Button(home, highlightthickness=0, bd=0, activebackground="white", image=img4, command=analyse)
b1.place(x=71, y=438)

photo = Image.open(directory + "assets/4.png")
img5 = ImageTk.PhotoImage(photo)
b1 = Button(home, highlightthickness=0, bd=0, activebackground="white", image=img5, command=about)
b1.place(x=71, y=531)

photo = Image.open(directory + "assets/5.png")
img6 = ImageTk.PhotoImage(photo)
b1 = Button(home, highlightthickness=0, bd=0, activebackground="white", image=img6, command=Exit)
b1.place(x=385, y=531)

txt = scrolledtext.ScrolledText(home, undo=True, fg="White", bg="#5bb1fc", bd=0, font=('', 15), height=13, width=38)
txt.place(x=695, y=230)

home.mainloop()
