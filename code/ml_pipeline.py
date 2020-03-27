from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from thundersvm import SVC


pipeline = Pipeline(
    steps=[
        ("vect", TfidfVectorizer(
                # ... Tfidf parameters
            ),
        ),
        ("sample", SMOTETomek()),
        ("clf", SVC(
                # ... SVC parameters
            ),
        ),
    ],
    verbose=True,
)
