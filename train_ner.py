from argparse import Namespace
from tqdm import trange, tqdm
import json
from pprint import pprint

from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
import copy

from vabert.dataset import VADataset
from vabert.trainer import BaselineNERTrainer
from vabert.utils import calc_scores
import os
import typer
from pathlib import Path
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import operator
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def index(ls, ids):
	return operator.itemgetter(*list(ids))(ls)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # specify which GPU(s) to be used

# TODO: handle input arguments
# TODO: re-train with ugaray96/biobert_ncbi_disease_ner (https://huggingface.co/ugaray96/biobert_ncbi_disease_ner)

def main(args):
	# Load and handling arguments 
	args_inp = json.load(open(args.config, 'r'))	# default parameters
	keys = list(args.__dict__.keys())				# overriding ones
	for key in keys:
		args.__dict__[key.upper()] = args.__dict__.pop(key)
		if args.__dict__[key.upper()]:
			args_inp[key.upper()] = args.__dict__[key.upper()]
	args = Namespace(**args_inp)
	pprint(args_inp)
                                                                                                                                                                 

	#  Load data, Tokenize and Pad 
	tokenizer = AutoTokenizer.from_pretrained(args.CHECKPOINT, do_lower_case=False)
	dataset = VADataset(args.DATA_FILE, tokenizer)    

	dataset.set_params(tokenizer=tokenizer,
	 				   max_len=args.MAX_LEN,
	 				   batch_size=args.BATCH_SIZE,
                       val_size=args.VAL_SIZE,
	 				   seed=args.SEED
    )

	train_ids, test_ids = train_test_split(range(len(dataset.sentences)))
	# Train (train and val)
	train_sents = index(dataset.sentences, train_ids);	train_tags = index(dataset.tags, train_ids)
	# Test
	test_sents = index(dataset.sentences, test_ids); 	test_tags = index(dataset.tags, test_ids)

 
	# Dataloaders
	train_dataloader = dataset.get_dataloaders(train_sents, train_tags, mode='all')
	valid_dataloader = dataset.get_dataloaders(test_sents, test_tags, mode='all')
	tag2idx, tag_values = dataset.get_tag_info()
	dataset.stats()
	print('--------------- DATA LOADED ----------------------')
 
	# Trainers
	trainer = BaselineNERTrainer()
	#args.DEVICE = torch.device('cuda:0')
	#torch.cuda.set_device(2)
	args.DEVICE = 'cuda'
	
	trainer.set_params(full_finetuning=args.FULL_FINETUNING,
                    	checkpoint=args.CHECKPOINT,
                     	max_epoch=args.MAX_EPOCH,
                      	max_grad_norm=args.MAX_GRAD_NORM,
                       	device=args.DEVICE)
	
 
	# --------------- TRAINING ------------------
	# Training assets
	model, optimizer, scheduler = trainer.setup_training(train_dataloader=train_dataloader, 
														tag2idx=tag2idx)
 
	## Store the average loss after each epoch so we can plot them.
	loss_values, validation_loss_values = [], []
	F1_best = 0

	# Tensorboard writer
	writer = SummaryWriter()

	for epoch in trange(args.MAX_EPOCH, desc="Epoch"):
		# ========================================
		#               Training
		# ========================================
		# Perform one full pass over the training set.

		# Training loop
		avg_train_loss, predictions_train, true_labels_train = trainer.epoch_train(model, train_dataloader, optimizer, scheduler)
		#print("Average train loss: {}".format(avg_train_loss))	
		
  		# Store the loss value for plotting the learning curve.
		loss_values.append(avg_train_loss)

		# calculate train scores
		P_train, R_train, F1_train = calc_scores(predictions_train, true_labels_train, tag_values, verbose=False)

		# ========================================
		#               Validation
		# ========================================
		# After the completion of each training epoch, measure our performance on
		# our validation set.
		avg_eval_loss, predictions_val, true_labels_val = trainer.epoch_validate(model, valid_dataloader)
		validation_loss_values.append(avg_eval_loss)
		#print("Validation loss: {}".format(avg_eval_loss))
		P_val, R_val, F1_val = calc_scores(predictions_val, true_labels_val, tag_values, verbose=False)

		# Save model
		if not os.path.exists('./models_temp'):
			os.makedirs('./models_temp')

		if F1_best < F1_val and F1_val > 0.7:
			tqdm.write('Improve F1-score from {:.4f} to {:.4f} at epoch {} | P: {:.4f} | R: {:.4f}'.format(F1_best, F1_val, epoch, P_val, R_val))
			F1_best = F1_val
			nonimproved_epoch = 0
			model.save_pretrained('models_temp/hponer_epoch{}_f1_{:.4f}'.format(epoch, F1_val)) # Make sure the folder `models/` exists
	  
		# Write to tensorboard
		writer.add_scalar('Loss/train', avg_train_loss, epoch)
		writer.add_scalar('Loss/val', avg_eval_loss, epoch)
		writer.add_scalar('F1/train', F1_train, epoch)
		writer.add_scalar('F1/val', F1_val, epoch)


if __name__ == '__main__':
	# Handling arguments
	parser = ArgumentParser()
	parser.add_argument('--config', 	type=str, default='config/train_config.json') # config's always required, all other argument will overdrive when requested
	parser.add_argument('--data_file', 	type=str)
	parser.add_argument('--checkpoint', type=str)
	parser.add_argument('--max_epoch', 	type=int)
	args = parser.parse_args()

	# Run and measure time
	from datetime import datetime 
	start_time = datetime.now() 

	main(args)

	time_elapsed = datetime.now() - start_time 
	print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))