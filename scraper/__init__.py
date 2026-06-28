from .find_sentenceids_by_lemmaid import download_tla_lemma_sentences
from .find_content_from_sentenceids import download_sentence_contents
from .find_urls_by_lemma import download_tla_lemma_sentences as download_tla_lemma_sentences_by_lemma
from .find_content_from_urls import download_sentence_contents as download_sentence_contents_by_urls

__all__ = [
    "download_tla_lemma_sentences",
    "download_sentence_contents",
    "download_tla_lemma_sentences_by_lemma",
    "download_sentence_contents_by_urls",
]
