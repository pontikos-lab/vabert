from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np 
import torch
from transformers import BatchEncoding
from vabert.utils import get_ents_from_bio, assign_entities, get_html
import streamlit as st

device = 'cpu'
checkpoint = "dmis-lab/biobert-v1.1"
checkpoint_weight = "models_temp/hponer_epoch280_f1_0.9561/pytorch_model.bin"
colors = {'VA': 'red', 'Laterality': 'blue', 'Vision': 'green'}


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_assets(device, checkpoint_weight, checkpoint):
	"""
	Load main assets for predictions (model, bert tokenizer, spacy tokenizer, tag dictionary)
	"""
	print('>>> Load assets ...')
	tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case=False)

	tag_values = ['O', 'B-VA', 'I-VA', 'B-Vision', 'I-Vision', 'B-Laterality', 'I-Laterality', 'B-Pinhole', 'I-Pinhole'] # TODO: bad handling, need optimization
	tag_values.append("PAD")
	tag2idx = {t: i for i, t in enumerate(tag_values)}

	model = AutoModelForTokenClassification.from_pretrained(checkpoint)
	model.classifier = torch.nn.Linear(768, len(tag2idx))
	model.num_labels = len(tag2idx)
	model.output_attentions = False
	model.output_hidden_states = False

	model.load_state_dict(torch.load(checkpoint_weight, map_location=torch.device(device)))
	return model, tag2idx, tag_values, tokenizer


def coloring(sentence, entities, colors):
	# Iterate over the entities and color the corresponding text in the sentence
	offset = 0
	for entity in entities:
		start = entity['start'] + offset
		end = entity['end'] + offset
		color = colors.get(entity['label'], 'black')
		sentence = sentence[:start] + f":{color}[{sentence[start:end]}]" + sentence[end:]
		offset += len(f":{color}[]")

	# Print the colored sentence
	return sentence


# Creating UI
st.title('VABERT')
test_sentence = st.text_area('')
st.caption('Ex:') 
st.caption('* Diagnosis: Right Full Thickness Macular Hole  Aided Vision: RE 6/60 LE 6/12.') 
st.caption("* Diagnosis: POAG, R cataract, L Phaco+IOL and Trabectome March 2013 VA's: R 6/6, L 6/4 IOP's R 21, L 17 mmHg")

# Preparing
model, tag2idx, tag_values, tokenizer, nlp = load_assets(device, checkpoint_weight, checkpoint)
tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cpu()

# Predict
if st.button('Run'):
	tokenized_batch : BatchEncoding = tokenizer(test_sentence)
	input_ids = torch.tensor([tokenized_batch['input_ids']]).cpu()

	# predict
	model.eval()
	with torch.no_grad():
		output = model(input_ids)
	label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

	# process result
	tokens = tokenized_batch[0].tokens
	new_tokens, new_labels = [], []
	for token, label_idx in zip(tokens, label_indices[0]):
		if token not in ['[CLS]', '[SEP]', '[PAD]']:
			new_labels.append(tag_values[label_idx])
			new_tokens.append(token)
	spans = get_ents_from_bio(tokenized_batch, new_labels, test_sentence)

	#  Display
	colored_sent =  coloring(test_sentence, spans, colors)
	st.markdown()
