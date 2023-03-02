import pandas as pd
from transformers import BertForTokenClassification, AdamW, RobertaForTokenClassification, AutoModelForTokenClassification
from transformers import get_linear_schedule_with_warmup
import torch
import numpy as np

class BaselineNERTrainer:
	def __init__(self):
		pass
	
	def set_params(self, full_finetuning, checkpoint, max_epoch, max_grad_norm, device):
		self.FULL_FINETUNING = full_finetuning
		self.MAX_EPOCH = max_epoch
		self.MAX_GRAD_NORM = max_grad_norm
		self.CHECKPOINT = checkpoint
		self.DEVICE = device
  
	def setup_training(self, train_dataloader, tag2idx):
		"""Setting up training objects
		
		Returns:
			model
			device
			optimizer
			scheduler
		"""
		# Load Model
		if self.FULL_FINETUNING: # TODO: re-assess this
			model = AutoModelForTokenClassification.from_pretrained(self.CHECKPOINT)
			model.classifier = torch.nn.Linear(768, len(tag2idx))
			model.num_labels = len(tag2idx)
			model.output_attentions = False
			model.output_hidden_states = False
		else:
			model = BertForTokenClassification.from_pretrained(
				self.CHECKPOINT,
				num_labels=len(tag2idx),
				output_attentions = False,
				output_hidden_states = False
			)

		if self.DEVICE == 'cuda':
			model.cuda()
		else:
			model.cpu()

		if self.FULL_FINETUNING:
			param_optimizer = list(model.named_parameters())
			no_decay = ['bias', 'gamma', 'beta']
			optimizer_grouped_parameters = [
				{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
				'weight_decay_rate': 0.01},
				{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
				'weight_decay_rate': 0.0}
			]
		else:
			param_optimizer = list(model.classifier.named_parameters())
			optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

		optimizer = AdamW(
			optimizer_grouped_parameters,
			lr=3e-5,
			eps=1e-8
		)

		# Total number of training steps is number of batches * number of epochs.
		total_steps = len(train_dataloader) * self.MAX_EPOCH

		# Create the learning rate scheduler.
		scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=0,
			num_training_steps=total_steps
		)
		return model, optimizer, scheduler

	def epoch_train(self, model, train_dataloader, optimizer, scheduler):
     	# Put the model into training mode.
		model.train()
		# Reset the total loss for this epoch.
		total_loss = 0

		predictions , true_labels = [], [] 	# for val scores
  
		for step, batch in enumerate(train_dataloader):
			# add batch to gpu
			batch = tuple(t.to(self.DEVICE) for t in batch)
			b_input_ids, b_input_mask, b_labels = batch
			# Always clear any previously calculated gradients before performing a backward pass.
			model.zero_grad()

			# forward pass
			# This will return the loss (rather than the model output)
			# because we have provided the `labels`.
			outputs = model(b_input_ids, token_type_ids=None,
							attention_mask=b_input_mask, labels=b_labels)
   
			# get the train loss
			loss = outputs[0]
			# Perform a backward pass to calculate the gradients.
			loss.backward()
			# track train loss
			total_loss += loss.item()
   
			# Clip the norm of the gradient
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.MAX_GRAD_NORM)
			# update parameters
			optimizer.step()
			# Update the learning rate.
			scheduler.step()

			# Calculate the average loss over the training data.
			avg_train_loss = total_loss / len(train_dataloader)

			# for for calculating train scores
			logits = outputs[1].detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			# for calculating the val scores
			predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
			true_labels.extend(label_ids)

			return avg_train_loss, predictions, true_labels

	def epoch_validate(self, model, valid_dataloader):
		# Put the model into evaluation mode
		model.eval()
		# Reset the validation loss for this epoch.
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		predictions , true_labels = [], [] 	# for val scores
		
		for batch in valid_dataloader:
			batch = tuple(t.to(self.DEVICE) for t in batch)
			b_input_ids, b_input_mask, b_labels = batch

			# Telling the model not to compute or store gradients,
			# saving memory and speeding up validation
			with torch.no_grad():
				# Forward pass, calculate logit predictions.
				# This will return the logits rather than the loss because we have not provided labels.
				outputs = model(b_input_ids, token_type_ids=None,
								attention_mask=b_input_mask, labels=b_labels)
    
			# Val loss
			eval_loss += outputs[0].mean().item()

			# Move logits and labels to CPU
			logits = outputs[1].detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			# for calculating the val scores
			predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
			true_labels.extend(label_ids)

		avg_eval_loss = eval_loss / len(valid_dataloader)

		return avg_eval_loss, predictions, true_labels