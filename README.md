# 🐶 Dog Breed Classifier

This project is a web app built using **Streamlit** and **TensorFlow** that can classify an image of a dog into one of 70 possible breeds. It uses a pre-trained **MobileNetV2** model fine-tuned on a custom dataset of labeled dog images.

---

## 🚀 Demo

Try the live app here: [https://dogapp-ugsv.onrender.com/](https://dogapp-ugsv.onrender.com/)

Upload a picture of a dog and get its predicted breed in real-time!

---

## 📂 Dataset

* The training and validation datasets are organized in folders:

  ```
  /train/<breed_name>/
  /valid/<breed_name>/
  ```
* There are **70 classes (breeds)**.
* Each folder contains images of dogs of that specific breed.

---

## 🧠 Model

* Base: [`MobileNetV2`](https://arxiv.org/abs/1801.04381) pretrained on ImageNet.
* Transfer learning with a custom classifier head:

  * GlobalAveragePooling → Dense → Softmax
* Trained using `categorical_crossentropy` loss and `Adam` optimizer.

---

## 🖼️ Web App Features

* Upload a dog image (JPG/PNG)
* Model predicts the breed
* Displays prediction with confidence score

---

## 📦 Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

### `requirements.txt` (example)

```txt
streamlit
tensorflow
numpy
pandas
Pillow
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                  # Streamlit app
├── dog_breed_model.h5      # Trained model
├── requirements.txt
├── README.md
├── /train/                 # Training dataset
└── /valid/                 # Validation dataset
```

---

## ✍️ Author

Made by [Anurag Pokhariyal](https://github.com/AnuragP004)

---

## 📜 License

This project is open source under the MIT License.
