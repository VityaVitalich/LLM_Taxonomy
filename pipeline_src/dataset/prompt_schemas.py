def clean_elem(elem, keys_to_remove_digits=["children"]):
    removes = set(keys_to_remove_digits)
    if not "changed" in elem.keys():
        for field in ["children", "parents", "grandparents", "brothers"]:
            if field in elem.keys():
                elem[field] = delete_techniqal(elem[field], remove=(field in removes))
                elem["changed"] = True
    return elem


def delete_techniqal(elem, remove):
    if isinstance(elem, str):
        if ".n." in elem and remove:
            return elem.split(".")[0].replace("_", " ")
        else:
            return elem.replace("_", " ")

    elif isinstance(elem, list):
        new_words = []
        for word in elem:
            new_words.append(delete_techniqal(word, remove))
        return new_words


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

    clean = clean_elem(elem, keys_to_remove_digits=["children"])
    # transformed_term = (
    #     ", ".join(clean["grandparents"])
    #     + " are hyponyms for the word '"
    #     + clean["parents"]
    #     + "'. Predict hyponyms for the word '"
    #     + clean["parents"]
    #     + "'. Answer:"
    # )
    transformed_term = (
        "hyperhypenyms: "
        + ", ".join(clean["grandparents"])
        + ", hypernym: "
        + clean["parents"]
        + " | hyponyms:"
    )
    return transformed_term, ", ".join(clean["children"])


def predict_child_from_parent(elem):
    """
    hypernym: elem['parents']
    hyponyms: {elem['children']}

    Predict hyponyms for the word "blackfly".  Answer:
    """

    clean = clean_elem(elem, keys_to_remove_digits=["children"])
    # transformed_term = (
    #     "Predict hyponyms for the word '" + clean["parents"] + "'.  Answer:"
    # )
    transformed_term = "hypernym: " + clean["parents"] + " | hyponyms:"
    return transformed_term, ", ".join(clean["children"])


def predict_children_with_parent_and_brothers(elem):
    """
    hypernym: elem['parents'],
    hyponyms: elem['brothers'], other hyponyms:

    Blackfly is hyponym for the word fly.
    Predict other hyponyms for the word “fly”. Answer:
    """

    clean = clean_elem(elem, keys_to_remove_digits=["children"])
    # transformed_term = (
    #     ", ".join(clean["brothers"])
    #     + "are hyponyms for the word '"
    #     + clean["parents"]
    #     + "'. Predict other hyponyms for the word '"
    #     + clean["parents"]
    #     + "'. Answer:"
    # )
    transformed_term = (
        "hypernym: "
        + clean["parents"]
        + ", hyponyms:"
        + ", ".join(clean["brothers"])
        + " | other hyponyms:"
    )
    return transformed_term, ", ".join(clean["children"])


def predict_child_from_2_parents(elem):
    """
    Predict common hyponyms for the words "cocker spaniel" and “poodle”.
    Answer:
    """
    clean = clean_elem(elem, keys_to_remove_digits=["children"])
    # transformed_term = (
    #     "Predict common hyponyms for the words '"
    #     + clean["parents"][0]
    #     + "' and '"
    #     + clean["parents"][1]
    #     + "'. Answer:"
    # )
    transformed_term = (
        "first hypernym: "
        + clean["parents"][0]
        + ", second hypernym: "
        + clean["parents"][1]
        + " | hyponyms:"
    )
    return transformed_term, clean["children"]


def predict_parent_from_child_granparent(elem):
    """
    Predict the hypernym for the word “spaniel” which is hyponyms for the
    word “hunting dog” at the same time. Answer: (sporting dog)
    """
    clean = clean_elem(elem, keys_to_remove_digits=["parents"])
    # transformed_term = (
    #     "Predict the hypernym for the word '"
    #     + clean["children"]
    #     + "' which is hyponyms for the word '"
    #     + clean["grandparents"]
    #     + "' at the same time. Answer:"
    # )
    transformed_term = (
        "hyperhypenym: "
        + clean["grandparents"]
        + ", hyponym: "
        + clean["children"]
        + " | hypernym:"
    )
    return transformed_term, clean["parents"]
