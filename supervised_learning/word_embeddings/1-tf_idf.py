#!/usr/bin/env python3
"""Still following that one tutorial
https://pages.github.rpi.edu/kuruzj/website_introml_rpi/notebooks/08-intro-nlp/03-scikit-learn-text.html
For reference"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Documentation I forgot"""
    tfid_vec = TfidfVectorizer(vocabulary=vocab)

    tfid_vec.fit(sentences)

    ret_array = tfid_vec.transform(sentences)

    # Ok when it comes to this return
    # I'm actually not sure if I'm supposed to transform it
    # Lemme check task 0
    #   No
    return (ret_array.toarray(), tfid_vec.get_feature_names_out())