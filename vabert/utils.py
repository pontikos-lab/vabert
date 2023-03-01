from ctypes import alignment
from seqeval.metrics import f1_score, precision_score, recall_score
from typing import List, Dict
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens import Span
import warnings
from termcolor import colored
import ujson
from pathlib import Path
import srsly
from copy import deepcopy
import os
import pandas as pd
import ast


# ============================================================
#                Label Studio
# ============================================================
def jsonl2labelstudio(input_file, output_file):
	"""Convert jsonl's NER file to labelstudio import file (json)
	"""
	all_samples = []
	letters = [l for l in srsly.read_jsonl(os.path.join(input_file))]
	data = {}
	sample = {}
	for letter in letters: # loop thr all letters (and annotation) in one epidoe
		data['text'] = letter['text']
		sample['data'] = data
		result_dict = {}
		result = []
		for span in letter['spans']:
			result.append(
				{
					"from_name":"label",
					"to_name":"text",
					"type":"labels",
					"value":{
						"start": span['start'],
						"end": span['end'],
						"labels":["pnt"]
					}
				}
			)
		result_dict['result'] = result
		sample['annotations'] = [result_dict]
		all_samples.append(deepcopy(sample))

	# Saving 
	data_written = srsly.json_dumps(all_samples, indent=2)

	with open(output_file, 'w') as f:
		f.write(data_written)

# ============================================================
#                Processing/Converting tokens
# ============================================================
def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """Convert normal tokens to bert-like tokens (subword) and preserve the BIO labels

    Args:
        sentence (List[str]): list of 'normal' tokens of the original sentence
        text_labels (List[str]]): list of corresponding BIO tags
        tokenizer ([type]): bert tokenizer

    Returns:
        tokenized_sentence (List[str]): bert-like tokens
        labels (List[str]): corresponding BIO tags
    """
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def bio_to_entity_tokens(inp_bio_seq: List[str]) -> List[Dict]:
    """
    Gets as an input a list of BIO tokens and returns the starting and end tokens of each span
    @return: The return should be a list of dictionary spans in the form of [{"token_start": x,"token_end":y,"label":""]
    """
    out_spans = []

    b_toks = sorted([i for i, t in enumerate(inp_bio_seq) if "B-" in t])  # Get the indexes of B tokens
    sequence_len = len(inp_bio_seq)
    for start_ent_tok_idx in b_toks:
        entity_type = inp_bio_seq[start_ent_tok_idx].split("-")[1]
        end_ent_tok_idx = start_ent_tok_idx + 1
        if start_ent_tok_idx + 1 < sequence_len:  # if it's not the last element in the sequence
            for next_token in inp_bio_seq[start_ent_tok_idx + 1:]:
                if next_token.split("-")[0] == "I" and next_token.split("-")[1] == entity_type:
                    end_ent_tok_idx += 1
                else:
                    break
        out_spans.append(dict(token_start=start_ent_tok_idx, token_end=end_ent_tok_idx - 1, label=entity_type))
    return out_spans

def character_annotations_to_spacy_doc(inp_annotation: Dict, inp_model) -> Doc:
    """
    Converts an input sentence annotated at the character level for NER to a spaCy doc object
    It assumes that the inp_annotation has:
        1. "text" field
        2. "spans" field with a list of NER annotations in the form of  {"start": <ch_idx>, "end": <ch_idx>,
        "label": <NER label name>}
    
    """
    text = inp_annotation["text"]  # extra
    doc = inp_model.make_doc(text)  # extra
    ents = []  # extra
    if "spans" in inp_annotation.keys():
        for entities_sentence in inp_annotation["spans"]:
            start = entities_sentence["start"]
            end = entities_sentence["end"]
            label = entities_sentence["label"]
            span = doc.char_span(start, end, label=label)
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character" \
                      f" span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n "
                warnings.warn(msg)
            else:
                if len(ents) > 0: # ensure no span duplication
                    if not is_span_exist(span, ents):
                        ents.append(span)
                else:
                    ents.append(span)
    doc.ents = ents
    return doc


def is_span_exist(span: Span, ls_spans: list) -> bool:
    """Check if the given span exists in the given list of spans
    """
    for _span in ls_spans:
        if _span == span:
            return True
    return False

def get_iob_labels(inp_doc: Doc) -> List[str]:
    """Convert a spacy doc (with ents) to list of corresponding BIO tags"""
    return [token.ent_iob_ + "-" + token.ent_type_ if token.ent_type_ else token.ent_iob_ for token in inp_doc]

def assign_entities(doc, token_spans):
    all_spans = []
    for span in token_spans:
        all_spans.append(Span(doc, span['token_start'], span['token_end']+1, span['label']))
        doc.set_ents(all_spans)
    return doc


# ============================================================
#                       I/O
# ============================================================
def write_jsonl(file_path, lines):
    # Taken from prodigy
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))


def read_jsonl(file_path):
    # Taken from prodigy support
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue

def parse_annfile(filepath):
    """Parse annotation file to anntations

    .jsonl: every line contains one dictionary in which there are two fields
        text: text of the sample
        spans: list of dictionaries, each dict is a mention, containing 
            "start": start location
            "end": end location
            "label": entity type
            "str": mention string
    .csv: exported csv file from LabelStudio, containing the columns: text,id,label,annotator,annotation_id,created_at,updated_at,lead_time
    .json: exporeted json file from LabelStudio
    """
    filename, ext = os.path.splitext(filepath)
    if ext == '.jsonl':
        _anns = srsly.read_jsonl(filepath)
        anns = [annotation for annotation in _anns]
    elif ext == '.csv':
        df = pd.read_csv(filepath)
        anns = []
        for index, row in df.iterrows():
            text = row['text']
            _id = row['id']
            spans = []
            for span in ast.literal_eval(row['label']): # example of a span  {""start"": 223, ""end"": 236, ""labels"": [""pnt""]}
                start = span['start']
                end = span['end']
                label = span['labels'][0]
                _str = text[start:end] # mention/span's string
                spans.append({"start": start,
                                "end": end,
                                "label":label,
                                "str": _str})
            anns.append({"text":text, "spans": spans, "id": _id})
    elif ext == '.json': #TODO: find a way to join the overlapping parts of csv and json
        _anns = srsly.read_json(filepath)
        anns = []
        for ann in _anns:
            text = ann['data']['text']
            _id = ann['id']
            spans = []
            for span in ann['annotations'][0]['result']: # example of a span  {""start"": 223, ""end"": 236, ""labels"": [""pnt""]}
                start = span['value']['start']
                end = span['value']['end']
                label = span['value']['labels'][0]
                _str = text[start:end] # mention/span's string
                spans.append({"start": start,
                                "end": end,
                                "label":label,
                                "str": _str})
            anns.append({"text":text, "spans": spans, "id": _id})
    return anns


# ============================================================
#                       Visualization
# ============================================================
def view_all_entities_terminal(inp_text: str, character_annotations: list):
    """Return text with colored entities that can be display on terminal

    Args:
        inp_text (str): sentence
        character_annotations (List[char-span]): Ex: [{'start': 51, 'end': 71, 'label': 'pnt'}, [{'start':...}]]
    """
    if character_annotations:
        character_annotations = sorted(character_annotations, key=lambda anno: anno['start'])
        sentence_text = ""
        end_previous = 0
        for annotation in character_annotations:
            sentence_text += inp_text[end_previous:annotation["start"]]
            sentence_text += colored(inp_text[annotation["start"]:annotation["end"]],
                                     'green', attrs=['reverse', 'bold'])
            end_previous = annotation["end"]
        sentence_text += inp_text[end_previous:]
        return sentence_text
    return inp_text

def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


# ============================================================
#                           Metrics
# ============================================================
def calc_scores(predictions: list, true_labels: list, tag_values, verbose): # TODO: add more details to input type, should be List[List[str]]?
    """Calculate precision, recall, and F1 of a batch of predictions

    Arguments:
        predictions (list): list of predictions from model (see more in hpo.trainer.baselineNERTrainer.epoch_validate())
        true_labels (list): list of true labels
        tag_values (dict): dictionary converting prediction values to BIO tags
    """
    pred_tags , valid_tags = [], []
    for i in range(len(predictions)):
        pred_tags.append([tag_values[p_i] for p_i, l_i in zip(predictions[i], true_labels[i]) if tag_values[l_i] != "PAD"])
        valid_tags.append([tag_values[l_i] for l_i in true_labels[i] if tag_values[l_i] != "PAD"])

    P = precision_score(pred_tags, valid_tags)	# precision
    R = recall_score(pred_tags, valid_tags, zero_division=1)		# recall
    F1 = f1_score(pred_tags, valid_tags)		# f1

    if verbose:
        print("P:  {}".format(P))
        print("R:  {}".format(R))
        print("F1: {}".format(F1))

    return P, R, F1
