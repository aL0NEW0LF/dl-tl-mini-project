import cv2
import numpy as np
import tensorflow


def load_image(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, (64, 64))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tensorflow.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


class Classifier:
    """
    Classifier class that handles image classification using a pre-trained Keras model.
    """

    def __init__(self, modelPath, labelsPath=None):
        """
        Initializes the Classifier with the model and labels.

        :param modelPath: str, path to the Keras model
        :param labelsPath: str, path to the labels file (optional)
        """
        self.model_path = modelPath
        np.set_printoptions(suppress=True)  # Disable scientific notation for clarity

        # Load the Keras model
        base_model = tensorflow.keras.applications.EfficientNetB7(
            input_shape=(64, 64, 3), include_top=False, weights=None
        )

        self.model = tensorflow.keras.models.Sequential(
            [
                base_model,
                tensorflow.keras.layers.GlobalAveragePooling2D(),
                tensorflow.keras.layers.Dropout(0.5),
                tensorflow.keras.layers.Dense(2560, activation="relu"),
                tensorflow.keras.layers.Dense(32, activation="softmax"),
            ]
        )

        self.model.load_weights(
            "D:\\Projects\\dl-tl-mini-project\\model\\model_checkpoint.keras"
        )

        # Create a NumPy array with the right shape to feed into the Keras model
        self.data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)

        self.labels_path = labelsPath

        # If a labels file is provided, read and store the labels
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = [line.strip() for line in label_file]
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        """
        Classifies the image and optionally draws the result on the image.

        :param img: image to classify
        :param draw: whether to draw the prediction on the image
        :param pos: position where to draw the text
        :param scale: font scale
        :param color: text color
        :return: list of predictions, index of the most likely prediction
        """
        # Resize and normalize the image
        self.data[0] = load_image(img)

        # Run inference
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        # Draw the prediction text on the image if specified
        if draw and self.labels_path:
            cv2.putText(
                img,
                str(self.list_labels[indexVal]),
                pos,
                cv2.FONT_HERSHEY_COMPLEX,
                scale,
                color,
                2,
            )

        return list(prediction[0]), indexVal
