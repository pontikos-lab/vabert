from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np 
import torch
from transformers import BatchEncoding
from vabert.utils import get_ents_from_bio

device = 'cpu'
checkpoint = "dmis-lab/biobert-v1.1"
checkpoint_weight = "models_temp/hponer_epoch280_f1_0.9561/pytorch_model.bin"


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
	print('>>> Done.')


def predict(test_sentence, model, tokenizer, tag_values):
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
	return spans

def process(test):
	model, tag2idx, tag_values, tokenizer = load_assets(device, checkpoint_weight, checkpoint)
	test = test.replace('\n\n\n', ' ').replace('\n\n', ' ').replace('\n', ' ').replace('   ', ' ').replace('  ',' ')
	test = test.replace(' .', '.')
	spans = predict(test, model, tokenizer, tag_values)
	return spans



if __name__ == '__main__':
	test_sentence = "Diagnosis: Right Full Thickness Macular Hole  Aided Vision: RE 6/60 LE 6/12."
	print(process(test_sentence))