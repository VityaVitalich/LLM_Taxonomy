def hypo_term_hyper(term):
    """
    hyponym: {term} | hypernyms:
    """
    transormed_term = "hyponym: " + term + " | hypernyms:"

    return transormed_term


def predict_child_with_parent_and_grandparent(elem):
    """
    hyperhypenym: arthropod.n.01,
    hypernym: insect.n.01, hyponyms:
    (blackly)

    hyperhypenym: elem['grandparents'],
    hypernym: elem['parents'], hyponyms:
    elem['children']

    Fly is a hyponym for the word “insect".
    Predict hyponyms for the word “fly”. Answer:
    """

    # transformed_term = (
    #     "hyperhypenym: "
    #     + ", ".join(elem["grandparents"])
    #     + ", hypernym: "
    #     + elem["parents"]
    #     + ", hyponyms:"
    # )
    transformed_term = (
        ", ".join(elem["grandparents"])
        + " are hyponyms for the word '"
        + elem["parents"]
        + "'. Predict hyponyms for the word '"
        + elem["parents"]
        + "'. Answer:"
    )
    return transformed_term


def predict_child_from_parent(elem):
    """
    hypernym: elem['parents']
    hyponyms: {elem['children']}

    Predict hyponyms for the word "blackfly".  Answer:
    """

    # transformed_term = "hypernym: " + elem["parents"] + ", hyponyms:"
    transformed_term = (
        "Predict hyponyms for the word '" + elem["parents"] + "'.  Answer:"
    )
    return transformed_term


def predict_children_with_parent_and_brothers(elem):
    """
    hypernym: elem['parents'],
    hyponyms: elem['brothers'], other hyponyms:

    Blackfly is hyponym for the word fly.
    Predict other hyponyms for the word “fly”. Answer:
    """

    # transformed_term = (
    #     "hypernym: "
    #     + elem["parents"]
    #     + ", hyponyms:"
    #     + ", ".join(elem["brothers"])
    #     + ", other hyponyms:"
    # )

    transformed_term = (
        ", ".join(elem["brothers"])
        + "are hyponyms for the word '"
        + elem["parents"]
        + "'. Predict other hyponyms for the word '"
        + elem["parents"]
        + "'. Answer:"
    )
    return transformed_term


def predict_child_from_2_parents(elem):
    """
    Predict common hyponyms for the words "cocker spaniel" and “poodle”.
    Answer:
    """
    transformed_term = (
        "Predict common hyponyms for the words '"
        + elem["parents"][0]
        + "' and '"
        + elem["parents"][1]
        + "'. Answer:"
    )
    return transformed_term


def predict_parent_from_child_granparent(elem):
    """
    Predict the hypernym for the word “spaniel” which is hyponyms for the
    word “hunting dog” at the same time. Answer: (sporting dog)
    """
    transformed_term = (
        "Predict the hypernym for the word '"
        + elem["children"]
        + "' which is hyponyms for the word '"
        + elem["grandparents"]
        + "' at the same time. Answer:"
    )
    return transformed_term
