# \# 🛢️ AI-Powered Oil Spill Detection using Satellite Imagery

# 

# \## 1. Overview 📜

# 

# Oil spills pose significant environmental and economic threats, demanding rapid detection and response. Traditional methods like manual satellite image inspection or physical patrols are often slow, costly, and inefficient. This project presents an AI-powered system leveraging deep learning to automatically detect and segment oil spills in satellite imagery, aiming for faster, more accurate identification to support mitigation efforts.

# 

# This repository contains the code for a web application built with Streamlit that allows users to upload satellite images and visualize the detected oil spill areas in real-time using a trained U-Net model with a pre-trained backbone.

# 

# ---

# 

# \## 2. Dataset 💾

# 

# \* \*\*Source:\*\* The model was trained on the "Annotated RGB images of Oil Spills in a Port Environment" dataset available on Zenodo: \[https://zenodo.org/records/10555314](https://zenodo.org/records/10555314).

# \* \*\*Type:\*\* The dataset consists of RGB aerial images (JPG format) of port environments, paired with corresponding binary pixel-level segmentation masks (PNG format) indicating the spill areas.

# \* \*\*Structure:\*\* Data was pre-organized into `train`, `validation` (`val`), and `test` directories.

# 

# ---

# 

# \## 3. Setup Instructions ⚙️

# 

# To run this application locally, follow these steps:

# 

# 1\.  \*\*Clone the Repository:\*\*

# &nbsp;   Clone this repository to your local machine.

# &nbsp;   ```bash

# &nbsp;   git clone \[https://github.com/AbhishekBhade/INFOSYS-\_INTERNSHIP-OIL\_SPILL\_DETECTION-.git](https://github.com/AbhishekBhade/INFOSYS-\_INTERNSHIP-OIL\_SPILL\_DETECTION-.git)

# &nbsp;   cd INFOSYS-\_INTERNSHIP-OIL\_SPILL\_DETECTION- # Adjust folder name if necessary

# &nbsp;   ```

# 2\.  \*\*Create a Virtual Environment (Recommended):\*\*

# &nbsp;   ```bash

# &nbsp;   python -m venv venv

# &nbsp;   source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`

# &nbsp;   ```

# 3\.  \*\*Install Dependencies:\*\*

# &nbsp;   Install all required Python libraries using the `requirements.txt` file:

# &nbsp;   ```bash

# &nbsp;   pip install -r requirements.txt

# &nbsp;   ```

# 4\.  \*\*Place Model File:\*\*

# &nbsp;   Ensure the trained model file (`best\_model\_final.keras`) is present in the main project directory alongside `app.py`.

# 

# ---

# 

# \## 4. Usage Instructions 🚀

# 

# 1\.  Navigate to the project directory in your terminal.

# 2\.  Run the Streamlit application:

# &nbsp;   ```bash

# &nbsp;   streamlit run app.py

# &nbsp;   ```

# 3\.  Open the provided local URL (e.g., `http://localhost:8501`) in your web browser.

# 4\.  Use the \*\*"Upload Satellite Image"\*\* button to select a `.jpg`, `.jpeg`, or `.png` file.

# 5\.  View the original image and the predicted oil spill overlay.

# 6\.  Use the controls in the \*\*⚙️ Analysis Controls\*\* sidebar to adjust:

# &nbsp;   \* \*\*Confidence Threshold:\*\* Controls detection sensitivity.

# &nbsp;   \* \*\*Overlay Transparency:\*\* Adjusts the visibility of the spill highlight.

# &nbsp;   \* \*\*Spill Highlight Color:\*\* Choose the color for the overlay.

# 7\.  Use the \*\*"Download Overlay Image"\*\* and \*\*"Download Mask Image"\*\* buttons to save the generated results.

# 

# ---

# 

# \## 5. Model Architecture 🧠

# 

# The core of this system is a \*\*U-Net\*\* architecture, implemented using the `segmentation-models` library. Key aspects include:

# 

# \* \*\*Encoder (Backbone):\*\* A pre-trained \*\*ResNet34\*\* model, initialized with weights learned from the large-scale ImageNet dataset (`encoder\_weights='imagenet'`), serves as the encoder. This \*\*transfer learning\*\* approach provides powerful, pre-learned feature extraction capabilities, crucial for achieving high accuracy with limited domain-specific data.

# \* \*\*Decoder:\*\* The decoder path uses transpose convolutional layers (`Conv2DTranspose`) to upsample the feature maps back to the original image resolution, progressively refining the segmentation.

# \* \*\*Skip Connections:\*\* Feature maps from corresponding encoder layers are concatenated with the decoder layers. This allows the model to combine high-level semantic information with low-level spatial details for precise boundary detection.

# \* \*\*Output Layer:\*\* A final 1x1 Convolution layer with a \*\*sigmoid activation\*\* produces a pixel-wise probability map (values between 0 and 1), representing the likelihood that each pixel belongs to an oil spill (`classes=1`).

# 

# ---

# 

# \## 6. Training Summary 📈

# 

# \* \*\*Framework:\*\* TensorFlow \& Keras, utilizing the `segmentation-models` library.

# \* \*\*Loss Function:\*\* \*\*Dice Loss\*\* (`sm.losses.DiceLoss()`). Chosen for its effectiveness in segmentation tasks, especially with potential class imbalances, as it directly optimizes for overlap (IoU).

# \* \*\*Evaluation Metric:\*\* \*\*Intersection over Union (IoU)\*\* Score (`sm.metrics.IOUScore()`) was the primary metric used for evaluating model performance during validation and testing. Accuracy was also monitored.

# \* \*\*Optimizer:\*\* Adam optimizer.

# \* \*\*Training Strategy:\*\* A \*\*two-phase fine-tuning\*\* approach was employed:

# &nbsp;   1.  \*\*Phase 1:\*\* Only the decoder layers of the U-Net were trained for 10 epochs while the pre-trained ResNet34 encoder remained frozen (its weights were not updated).

# &nbsp;   2.  \*\*Phase 2:\*\* All layers (encoder and decoder) were unfrozen and trained end-to-end with a significantly reduced learning rate (`1e-5`) to fine-tune the entire network on the specific task.

# \* \*\*Hyperparameter Tuning \& Regularization:\*\* Callbacks were used during Phase 2:

# &nbsp;   \* `EarlyStopping`: Monitored `val\_iou\_score` and stopped training if no improvement was seen for 5 epochs (`patience=5`), restoring the weights from the epoch with the best score.

# &nbsp;   \* `ModelCheckpoint`: Saved the model weights only when `val\_iou\_score` improved on the validation set.

# 

# ---

# 

# \## 7. Evaluation Results ✨

# 

# The model demonstrated high performance on the unseen test dataset:

# 

# \* \*\*Final Test IoU Score:\*\* \*\*0.9642 (96.4%)\*\*

# \* \*\*Final Test Accuracy:\*\* \*\*0.9654 (96.5%)\*\*

# \* \*\*Final Test Loss (Dice):\*\* 0.0184

# 

# \### Classification Report (Pixel-Level):

# 

# ```text

# &nbsp;                    precision    recall  f1-score   support

# 

# &nbsp;   Water (Class 0)       0.70      0.13      0.22    622123

# Oil Spill (Class 1)       0.97      1.00      0.98  16024021

# 

# &nbsp;          accuracy                           0.97  16646144

# &nbsp;         macro avg       0.83      0.56      0.60  16646144

# &nbsp;      weighted avg       0.96      0.97      0.95  16646144

# 

# ---

# 

# \## 8. Visual Output Examples 📸

# 

# The following images demonstrate the model's segmentation capabilities on sample test images.

# 

# \### Side-by-Side Comparison:

# (Input Image | Ground Truth Mask | Predicted Mask)

# 

# !\[Side-by-Side Example](images/side\_by\_side\_example.png)

# \*(Ensure `side\_by\_side\_example.png` is in the `images` folder)\*

# 

# \### Prediction Overlay:

# (Input Image with predicted spill highlighted)

# 

# !\[Overlay Example](images/overlay\_example.png)

# \*(Ensure `overlay\_example.png` is in the `images` folder)\*

# 

# ---

