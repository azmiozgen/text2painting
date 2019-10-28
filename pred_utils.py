import os


def get_model(usegpu):

    try:
        from models.script_identification_model.lib import Model, Prediction
        from models.script_identification_model.settings import ModelSettings, TrainingSettings
    except ImportError:
        from .models.script_identification_model.lib import Model, Prediction
        from .models.script_identification_model.settings import ModelSettings, TrainingSettings

    model_path = os.path.abspath(os.path.join(__file__, os.path.pardir, 'models', 'script_identification_model', 
                                                                        'script_identification_model.pth'))

    ms = ModelSettings()
    ts = TrainingSettings()

    model = Model(ts.MODEL, ts.N_CLASSES, load_model_path=model_path, usegpu=usegpu)
    prediction_model = Prediction(ms.IMAGE_SIZE_HEIGHT, ms.IMAGE_SIZE_WIDTH, ms.MEAN, ms.STD, model)

    return prediction_model

def identify_script(script_image_path, prediction_model, is_raw=False):

    try:
        from models.script_identification_model.settings import TrainingSettings
    except ImportError:
        from .models.script_identification_model.settings import TrainingSettings

    ts = TrainingSettings()

    _, _, pred, prob = prediction_model.predict(script_image_path, ts.LABELS, is_raw=is_raw)

    output = dict()
    output['Script'] = {'prediction' : str(pred), 'probability' : str(prob)}

    return output
