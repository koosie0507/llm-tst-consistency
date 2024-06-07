from pathlib import Path

import lftk
import pandas as pd
from spacy.tokens import Doc


class HandcraftedLinguisticFeature:
    def __init__(self, name: str):
        self._name = name

    def __call__(self, doc: Doc) -> float:
        extractor = lftk.Extractor(docs=doc)
        return extractor.extract(features=[self._name])[self._name]


class KupermanAgeOfAcquisition(HandcraftedLinguisticFeature):
    TABLE_PATH = Path(__file__).parent / "kup-aoa-ratings.csv"

    def __init__(self, name: str):
        super().__init__(name)
        self._aoa_ratings = pd.read_csv(self.TABLE_PATH)

    def __call__(self, doc: Doc) -> float:
        unique_words = set(token.lemma_.lower() for token in doc)
        aoa_of_unique_words = [
            round(rating)
            for word in unique_words
            for rating in self._aoa_ratings.loc[self._aoa_ratings["Word"] == word, "Rating.Mean"]
        ]
        if len(aoa_of_unique_words) == 0:
            return 100
        return round(sum(aoa_of_unique_words) / len(aoa_of_unique_words), 2)