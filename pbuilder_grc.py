
# coding: utf-8

import spacy
from spacy import displacy
import stanza
from spacy_stanza import StanzaLanguage
import streamlit as st
from spacy_streamlit import visualize_parser
import random
import numpy
from collections import Counter
from spacy.matcher import DependencyMatcher
from collections import defaultdict
from typing import Dict, List
from spacy.pipeline import merge_entities
import csv
from pathlib import Path

stanza.download('grc')

st.title("Diogenet's Greek Pattern Builder")



#st.sidebar.title("Model name")
#spacy_model = st.sidebar.selectbox("", ["en_core_web_lg", "en_diogenet_lg", "grc_diogenet_lg"])

snlp = stanza.Pipeline(lang="grc")
nlp = StanzaLanguage(snlp)


#nlp = spacy.load(spacy_model)
#nlp.add_pipe(merge_entities)


def visualise_doc(doc):
    #displacy.render(doc, style="dep", options={"distance": 120}, jupyter=True)
    #displacy.render(doc, style="ent", options={"distance": 120}, jupyter=True)
    options = {"compact": True, "add_lemma": True, "collapse_phrase": True}
    displacy.render(doc, style="dep", options=options)
    displacy.render(doc, style="ent")



def visualise_subtrees(doc, subtrees):

    words = [{"text": t.text, "tag": t.pos_} for t in doc]

    if not isinstance(subtrees[0], list):
        subtrees = [subtrees]

    for subtree in subtrees:
        arcs = []

        tree_indices = set(subtree)
        for index in subtree:

            token = doc[index]
            head = token.head
            if token.head.i == token.i or token.head.i not in tree_indices:
                continue

            else:
                if token.i < head.i:
                    arcs.append(
                        {
                            "start": token.i,
                            "end": head.i,
                            "label": token.dep_,
                            "dir": "left",
                        }
                    )
                else:
                    arcs.append(
                        {
                            "start": head.i,
                            "end": token.i,
                            "label": token.dep_,
                            "dir": "right",
                        }
                    )
        print("Subtree: ", subtree)
        # displacy.render(
        displacy.serve(
            {"words": words, "arcs": arcs},
            style="dep",
            options={"distance": 120},
            manual=True,
            #jupyter=True
        )

PTB_BRACKETS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
}

def clean_and_parse(sent: str, nlp):

    tokens = sent.strip().split(" ")

    new = []

    for token in tokens:
        new_token = PTB_BRACKETS.get(token, None)
        if new_token is None:
            new.append(token)
        else:
            new.append(new_token)

    return nlp(" ".join(new))


def parse_dep_path(dep_string: str):

    rules = [rule.split("|") for rule in dep_string.split(" ")]

    for triple in rules:

        if triple[0] in PTB_BRACKETS:
            triple[0] = PTB_BRACKETS[triple[0]]

        if triple[2] in PTB_BRACKETS:
            triple[2] = PTB_BRACKETS[triple[2]]

        if triple[1] == "nsubj:xsubj":
            triple[1] = "nsubj"

        if triple[1] == "nsubjpass:xsubj":
            triple[1] = "nsubjpass"
    return rules


def check_for_non_trees(rules: List[List[str]]):

    #print("reglas", rules)
    parent_to_children = defaultdict(list)
    seen = set()
    has_incoming_edges = set()
    for (parent, rel, child) in rules:
        #print(parent, rel, child)
        seen.add(parent)
        seen.add(child)
        has_incoming_edges.add(child)
        if parent == child:
            #print(parent, child)
            return None
        parent_to_children[parent].append((rel, child))
    print("ramas ", parent_to_children)

    # Only accept strictly connected trees.
    roots = seen.difference(has_incoming_edges)
    #print("raices ", roots)
    if len(roots) != 1:
        return None

    root = roots.pop()
    seen = {root}

    # Step 2: check that the tree doesn't have a loop:
    def contains_loop(node):
        has_loop = False
        #print("nodo ", parent_to_children[node])
        for (_, child) in parent_to_children[node]:
            if child in seen:
                print("has a loop", child)
                return True
            else:
                seen.add(child)
                has_loop = contains_loop(child)
            if has_loop:
                #print("had a loop")
                break

        return has_loop

    if contains_loop(root):
        return None

    return root, parent_to_children


def construct_pattern(rules: List[List[str]]):
    """
    Idea: add patterns to a matcher designed to find a subtree in a spacy dependency tree.
    Rules are strictly of the form "CHILD --rel--> PARENT". To build this up, we add rules
    in DFS order, so that the parent nodes have already been added to the dict for each child
    we encounter.
    """
    # Step 1: Build up a dictionary mapping parents to their children
    # in the dependency subtree. Whilst we do this, we check that there is
    # a single node which has only outgoing edges.

    if "dep" in {rule[1] for rule in rules}:
        return None

    ret = check_for_non_trees(rules)

    if ret is None:
        return None
    else:
        root, parent_to_children = ret

    def add_node(parent: str, pattern: List):

        for (rel, child) in parent_to_children[parent]:

            # First, we add the specification that we are looking for
            # an edge which connects the child to the parent.
            node = {
                "SPEC": {
                    "NODE_NAME": child,
                    "NBOR_RELOP": ">",
                    "NBOR_NAME": parent},
            }

            # DANGER we can only have these options IF we also match ORTH below, otherwise it's torturously slow.
            # token_pattern = {"DEP": {"IN": ["amod", "compound"]}}

            # Now, we specify what attributes we want this _token_
            # to have - in this case, we want to match a certain dependency
            # relation specifically.
            token_pattern = {"DEP": rel}

            # Additionally, we can specify more token attributes. So here,
            # if the node refers to the start or end entity, we require that
            # the word is part of an entity (spacy syntax is funny for this)
            # and that the word is a noun, as there are some verbs annotated as "entities" in medmentions.

            if child in {"START_ENTITY", "END_ENTITY"}:
                token_pattern["ENT_TYPE"] = {"NOT_IN": [""]}
                token_pattern["POS"] = "NOUN"
            elif child in {"PERSON_1", "PERSON_2", "PERSON_3", "GPE_1", "GPE_2", "ANY_1", "ANY_2"}:
                print(child)
                #token_pattern["ENT_TYPE"] = {"NOT_IN": [""]}
                #token_pattern["POS"] = "NOUN"
                #token_pattern["TEXT"] = {"REGEX": "*"}
            # If we are on part of the path which is not the start/end entity,
            # we want the word to match. This could be made very flexible, e.g matching
            # verbs instead, etc.
            else:
                token_pattern["LEMMA"] = child

            node["PATTERN"] = token_pattern

            pattern.append(node)
            add_node(child, pattern)

    pattern = [{"SPEC": {"NODE_NAME": root}, "PATTERN": {"lEMMA": root}}]
    # pattern = [{"SPEC": {"NODE_NAME": root}, "PATTERN": {"LEMMA": root}}] # to use lemmas as root
    add_node(root, pattern)

    assert len(pattern) < 20
    return pattern




#doc = nlp("Polycrates sent him on to those of Memphis, on the pretense that the were the more ancient.")
#doc = nlp("Pythagoras went to Delos")
st.header("Text to analyze:")
text = st.text_area("", "πλέοντος δὲ τοῦ Μνησάρχου εἰς τὴν Ἰταλίαν.")
doc = nlp(text) 


#sentence_spans = list(doc.sents)
#displacy.render(sentence_spans, style = "dep")
#visualise_doc(doc)
visualize_parser(doc)



#patterns = ["go|nsubj|ENTITY_ONE go|prep|to to|pobj|ENTITY_TWO"]
#patterns = ["send|nsubj|ENTITY_TWO send|dobj|ENTITY_THREE send|prep|at at|pobj|ENTITY_ONE"]

#patterns = []
#patterns = st.text_area("Pattern","["end|nsubj|ENTITY_TWO send|dobj|ENTITY_THREE send|prep|at at|pobj|ENTITY_ONE"]"])
st.header('Pattern:')
patterns_text = st.text_area("" ,"πλέω|nsubj|PERSON_1 πλέω|obl|GPE_1")
patterns = [patterns_text] 


matcher = DependencyMatcher(nlp.vocab)

stream = [doc]

count = 0
for pattern in patterns:
    rules = [rule.split("|") for rule in pattern.split(" ")]
    constructed_pattern = construct_pattern(rules)
    count += 1
    print("Adding these constructed patterns ", constructed_pattern)
    matcher.add("patron "+str(count), None, constructed_pattern)
         
    #print("Matcher full settings >")
    #print("patterns: ", matcher._patterns)
    #print("keys to token", matcher._keys_to_token)
    #print("root ", matcher._root)
    #print("entities ", matcher._entities)
    #print("callbacks ", matcher._callbacks)
    #print("nodes ", matcher._nodes)
    #print("tree ", matcher._tree)
    #print("< Matcher full settings")




# salida csv
# based on https://stackoverflow.com/questions/33309436/python-elementtree-xml-output-to-csv
with open('all_relations.csv', 'w', newline='') as r:  
    writer = csv.writer(r,  delimiter=' ', quotechar='"', quoting=csv.QUOTE_ALL)
    #writer.writerow(['id', 'relation','subject','to','destination', 'source_text'])  # WRITING HEADERS
    # rows vary in lenght. Therefore cannot use just one header
    counter = 1;
    for doc in stream:
        matches = matcher(doc) # do the matching
        #print(matches)
   
        relations = []
        logical_relations = []
        for match_id, token_idxs in matches:
           for each_pattern in token_idxs: 
               tokens = [doc[i] for i in each_pattern]
               heads = [doc[i].head for i in each_pattern]
               deps = [doc[i].dep_ for i in each_pattern]
               print("tokens>")
               print(tokens)
               print(heads)
               print(deps)
               print("<")
               one_row = [counter] + tokens 
               writer.writerow(one_row) # to the csv
               counter += 1
               
               logical_relations.append({"logic":str(tokens).strip('[]')}) # adding the logical relation to the db too
             
               branch = matcher._tree[match_id] # realnente es una lista de branches de este id
               print(branch)
               
               for k in branch[0]:
                   for rel, j in branch[0][k]:
                       st.write(tokens[k],"--",deps[j],rel,tokens[j])
                       relations.append({"child": int(each_pattern[j]), "head": int(each_pattern[k]), "label": deps[j]})             

  #          print(relations) 
        #print(logical_relations)
        st.write(logical_relations)
        #st.write(relations)






