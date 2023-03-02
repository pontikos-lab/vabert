from typing import List
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from vabert.utils import tokenize_and_preserve_labels, view_all_entities_terminal, parse_annfile

import spacy
import srsly
from vabert.utils import character_annotations_to_spacy_doc, get_iob_labels


# TODO: Add docstring to functions

class VADataset(Dataset):
	def __init__(self, filename:str) -> None:
		anns = parse_annfile(filename)
		nlp = spacy.load('tokenizers/super-tokenizer')

		self.sentences = []
		self.tags = []
		for ann in anns:
			try:
				doc = character_annotations_to_spacy_doc(ann, nlp)
			except: # when spans overlapped # TODO: handle this
				continue
			bio = get_iob_labels(doc)
			self.sentences.append([str(word) for word in doc])
			self.tags.append(bio)

		self.tag_values = ['O', 'B-VA', 'I-VA', 'B-Vision', 'I-Vision', 'B-Laterality', 'I-Laterality', 'B-Pinhole', 'I-Pinhole'] # TODO: bad handling, need optimization
		self.tag_values.append("PAD")
		self.tag2idx = {t: i for i, t in enumerate(self.tag_values)}
		self.anns = anns # this format is useful for some function (e.g. viewing entities in terminal)

	def __len__(self) -> int:
		assert len(self.sentences) == len(self.tags)
		return len(self.sentences)

	def __getitem__(self, index: int):
		return self.sentences[index], self.tags[index]

	def set_params(self, tokenizer, max_len, batch_size, val_size, seed):
		"""Set parameters for the Dataset
		Note: this function must be called before others

		Args:
			val_size (float): portion of validation set [0 - 1]
			seed (int): seed for randomization
			batch_size (int): batch size
			max_len (int): maximal length of the input vector (affecting padding)
			tokenizer (BertTokenizer): BERT tokenizer
		"""
		self.tokenizer = tokenizer
		self.VAL_SIZE = val_size
		self.SEED = seed
		self.BATCH_SIZE = batch_size
		self.MAX_LEN = max_len
		self.TOKENIZER = tokenizer

	def get_dataloaders(self, sents, tags, mode='split'):
		"""Return train and val dataloaders
		split -> tokenize -> pad -> [Dataloaders]

		mode [str]: 'split' or 'all' # TODO: add details
		"""
		if mode == 'split':
			# Split
			train_sents, valid_sents, train_tags, valid_tags = train_test_split(sents, tags,
																	random_state=self.SEED,
																	test_size=self.VAL_SIZE)

			# make dataloaders
			tokenized_train_sents, tokenized_train_tags, train_dataloader = self.make_dataloader(train_sents, train_tags)
			tokenized_valid_sents, tokenized_valid_tags, valid_dataloader = self.make_dataloader(valid_sents, valid_tags)

			# for later use
			self.train_sents = train_sents; self.train_tags = train_tags
			self.valid_sents = valid_sents; self.valid_tags = valid_tags
			self.tokenized_train_sents = tokenized_train_sents; self.tokenized_train_tags = tokenized_train_tags
			self.tokenized_valid_sents = tokenized_valid_sents; self.tokenized_valid_tags = tokenized_valid_tags
			return train_dataloader, valid_dataloader

		elif mode == 'all':
			_, _, dataloader = self.make_dataloader(sents, tags)
			return dataloader
		else:
			print('Error: please select mode') # TODO: Add raising error here

	def make_dataloader(self, sents, tags):
		# Tokenizing
		tokenized_sents, tokenized_tags = self._tokenize(sents, tags)

		# Padding
		inputs, tags_pad, masks = self._pad(tokenized_sents, tokenized_tags)

  		# Tensorizing
		inputs_ts = torch.tensor(inputs)
		tags_ts = torch.tensor(tags_pad)
		masks_ts = torch.tensor(masks)

		# Dataloaders
		data = TensorDataset(inputs_ts, masks_ts, tags_ts)
		sampler = RandomSampler(data)
		dataloader = DataLoader(data, sampler=sampler, batch_size=self.BATCH_SIZE)

		return tokenized_sents, tokenized_tags, dataloader


	def get_splitted(self):
		return  (self.train_sents, self.train_tags), (self.valid_sents, self.valid_tags)

	def get_tokenized(self):
		"""Get tokenized sentences and corresponding tags from train and validation set
		Prerequisite: get_dataloaders()

		Returns:
			tokenized train set (Tuple): train sentences and corresponding tags
   			tokenized valid set (Tuple): valid sentences and corresponding tags
		"""
		return (self.train_sents, self.train_tags), (self.valid_sents, self.valid_tags)

	def get_tag_info(self):
		return self.tag2idx, self.tag_values

	def stats(self, verbose=False):
		"""Get basics statistics of the dataset
		"""
		num_sent_has_ent = 0
		num_ent = 0
		for i in range(len(self.tags)):
			tag = self.tags[i]
			sent = self.sentences[i]
			if set(tag) != set('O'):	# sentence with all Os will have set(tag) == {'O'}
				num_sent_has_ent += 1
				num_ent += tag.count('B-pnt')

		if verbose:
			if set(tag) != set('O'):
				print(tag)
				print(tag.count('B-pnt'), ' | ', num_ent)
			else:
				print('#####', tag)
			print()

		print('Total sentences: ', len(self.sentences))
		print('No. of sentences that have entities: ', num_sent_has_ent)
		print('No. of sentences that have no entities (all Os): ', len(self.sentences) - num_sent_has_ent)
		print('No. of entities: ', num_ent)
		print('No. of entities/sentences: ', num_ent/num_sent_has_ent)

	def view_ents(self, index=None):
		if index is not None:
			sent = self.anns[index]
			print(view_all_entities_terminal(sent['text'], sent['spans']))
		else:
			for sent in self.anns: # show all
				print(view_all_entities_terminal(sent['text'], sent['spans']))

	def _tokenize(self, sents: List[List], tags: List[List], tokenizer=None):
		"""Tokennize the sentences

		Args:
			sents (List[List]): Sentences to be tokenized
			tags (List{List]): Corresponding tags
			tokenizer: tokenizer, eg BertTokenizer

		Returns:
			[type]: [description]
		"""
		if tokenizer is None:
			tokenizer = self.tokenizer

		tokenized_texts_and_labels = [
			tokenize_and_preserve_labels(sent, labs, tokenizer)
			for sent, labs in zip(sents, tags)
		]

		tokenized_sents = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
		tokenized_tags = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
		return tokenized_sents, tokenized_tags

	def _pad(self, tokenized_sents: List[List], tokenized_tags: List[List], max_len=None):
		"""Padding

		Args:
			tokenized_sents (List[List]): list of tokenized sentences
			tokenized_tags (List[List]): list of corresponding tags
			max_len (int): maximum length of the vector after padded

		Returns:
			input_id:
			tags:
			attention_maks:
		"""
		if max_len is None:
			max_len = self.MAX_LEN

		input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(sent) for sent in tokenized_sents],
								maxlen=max_len, dtype="long", value=0.0,
								truncating="post", padding="post")

		tags = pad_sequences([[self.tag2idx.get(t) for t in tag] for tag in tokenized_tags],
							maxlen=max_len, value=self.tag2idx["PAD"], padding="post",
							dtype="long", truncating="post")

		attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
		return input_ids, tags, attention_masks


if __name__ == '__main__':
	import json
	from transformers import BertTokenizer
	from argparse import Namespace

	params = json.load(open('config/train_config.json', 'r'))
	p = Namespace(**params)

	# Load data
	dataset = VADataset(p.DATA_FILE)

	#  Tokenize and Pad
	tokenizer = BertTokenizer.from_pretrained(p.CHECKPOINT, do_lower_case=False)
	dataset.set_params(tokenizer=tokenizer,
						max_len=p.MAX_LEN,
						batch_size=p.BATCH_SIZE,
						val_size=p.VAL_SIZE,
						seed=p.SEED
	)

	# Dataloaders
	train_dataloader, valid_dataloader = dataset.get_dataloaders()
