import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Import these for handling images

user_inputs = {}
values_list = []

def submit_form():
    pregnancies = entry_pregnancies.get()
    glucose = entry_glucose.get()
    blood_pressure = entry_blood_pressure.get()
    skin_thickness = entry_skin_thickness.get()
    insulin = entry_insulin.get()
    bmi = entry_bmi.get()
    diabetes_pedigree_function = entry_diabetes_pedigree_function.get()
    age = entry_age.get()

    user_inputs['pregnancies'] = pregnancies
    user_inputs['glucose'] = glucose
    user_inputs['blood_pressure'] = blood_pressure
    user_inputs['skin_thickness'] = skin_thickness
    user_inputs['insulin'] = insulin
    user_inputs['bmi'] = bmi
    user_inputs['diabetes_pedigree_function'] = diabetes_pedigree_function
    user_inputs['age'] = age

    # Convert values to a list of floats
    global values_list
    values_list = [float(value) for value in user_inputs.values()]
    window.destroy()

window = tk.Tk()
window.title("User Input Form")

# Set the window geometry to full screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window.geometry(f"{screen_width}x{screen_height}")

# Load and resize the background image
bg_image = Image.open("pexels-n-voitkevich-6941884.jpg")
bg_image = bg_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
bg_image = ImageTk.PhotoImage(bg_image)

# Create a Label widget to hold the image as the background
background_label = tk.Label(window, image=bg_image)
background_label.place(relwidth=1, relheight=1)  # Place it as a background

# Create a style for ttk widgets
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12))
style.configure("TButton", font=("Arial", 12))

label_pregnancies = ttk.Label(window, text="Please enter the number of pregnancies:")
label_pregnancies.pack(pady=10)
entry_pregnancies = ttk.Entry(window, font=("Arial", 12))
entry_pregnancies.pack(pady=10)

label_glucose = ttk.Label(window, text="Please enter the glucose level:")
label_glucose.pack(pady=10)
entry_glucose = ttk.Entry(window, font=("Arial", 12))
entry_glucose.pack(pady=10)

label_blood_pressure = ttk.Label(window, text="Please enter the blood pressure:")
label_blood_pressure.pack(pady=10)
entry_blood_pressure = ttk.Entry(window, font=("Arial", 12))
entry_blood_pressure.pack(pady=10)

label_skin_thickness = ttk.Label(window, text="Please enter the skin thickness:")
label_skin_thickness.pack(pady=10)
entry_skin_thickness = ttk.Entry(window, font=("Arial", 12))
entry_skin_thickness.pack(pady=10)

label_insulin = ttk.Label(window, text="Please enter the insulin level:")
label_insulin.pack(pady=10)
entry_insulin = ttk.Entry(window, font=("Arial", 12))
entry_insulin.pack(pady=10)

label_bmi = ttk.Label(window, text="Please enter the BMI (Body Mass Index):")
label_bmi.pack(pady=10)
entry_bmi = ttk.Entry(window, font=("Arial", 12))
entry_bmi.pack(pady=10)

label_diabetes_pedigree_function = ttk.Label(window, text="Please enter the diabetes pedigree function:")
label_diabetes_pedigree_function.pack(pady=10)
entry_diabetes_pedigree_function = ttk.Entry(window, font=("Arial", 12))
entry_diabetes_pedigree_function.pack(pady=10)

label_age = ttk.Label(window, text="Please enter your age:")
label_age.pack(pady=10)
entry_age = ttk.Entry(window, font=("Arial", 12))
entry_age.pack(pady=10)

button_submit = ttk.Button(window, text="Submit", command=submit_form)
button_submit.pack(pady=10)

window.mainloop()
