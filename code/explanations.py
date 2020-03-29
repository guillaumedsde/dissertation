from lime.lime_text import LimeTextExplainer
from harpocrates_server.service import CLASS_NAMES

MAX_FEATURES = 20

def lime_explanation(classifier, data, features=MAX_FEATURES):
    explainer = LimeTextExplainer(class_names=CLASS_NAMES)
    explanation = explainer.explain_instance(
        text_instance=data,
        classifier_fn=classifier.predict_proba,
        num_features=features,
    )
    return explanation