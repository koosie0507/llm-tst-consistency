import lftk
from spacy.tokens import Doc


class HandcraftedLinguisticFeature:
    def __init__(self, name: str):
        self._name = name

    def __call__(self, doc: Doc) -> float:
        extractor = lftk.Extractor(docs=doc)
        return extractor.extract(features=[self._name])[self._name]