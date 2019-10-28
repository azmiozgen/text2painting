from .utils import ImageUtilities


class Prediction(object):

    def __init__(self, resize_height, resize_width, mean, std, model):

        self.resize_height = resize_height
        self.resize_width = resize_width

        self.resizer = ImageUtilities.image_resizer(resize_height, resize_width)
        self.normalizer = ImageUtilities.image_normalizer(mean, std)
        self.model = model

    def preprocess(self, image_path, is_raw=False):
        img = ImageUtilities.read_image(image_path, is_raw=is_raw)

        img = self.resizer(img)
        img = self.normalizer(img)
        img = img.unsqueeze(0)

        return img

    def forward(self, image):

        prediction = self.model.predict(image)

        return prediction

    def postprocess(self, prediction_probs, labels):

        prediction_probs = prediction_probs.squeeze(0)
        prediction = int(prediction_probs.argmax(0))
        prediction = labels[prediction]
        prob = round(float(prediction_probs.max(0)), 5)

        return prediction, prob

    def predict(self, image_path, labels, is_raw=False):

        images = self.preprocess(image_path, is_raw=is_raw)
        predictions = self.forward(images)
        pred, prob = self.postprocess(predictions, labels)

        images_np = images.data.cpu().numpy()

        return images_np, predictions, pred, prob
