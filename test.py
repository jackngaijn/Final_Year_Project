import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

# Step 1: Load the SQuAD dataset
squad = load_dataset("squad")

# Extract data
train_data = squad['train']
contexts = [item['context'] for item in train_data]
questions = [item['question'] for item in train_data]
answers = [item['answers']['text'][0] for item in train_data]
answer_starts = [item['answers']['answer_start'][0] for item in train_data]

# Step 2: Preprocess the data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize context and question
def preprocess_data(context, question, answer, answer_start):
    tokenized_context = tokenizer(context, truncation=True, padding="max_length", max_length=300, return_tensors="pt")
    tokenized_question = tokenizer(question, truncation=True, padding="max_length", max_length=50, return_tensors="pt")
    
    # Compute answer tokens
    answer_end = answer_start + len(answer)
    token_start = tokenized_context.char_to_token(answer_start)
    token_end = tokenized_context.char_to_token(answer_end - 1)
    
    if token_start is None or token_end is None:
        # If the answer cannot be tokenized, return None (to filter out later)
        return None
    
    return {
        "context": tokenized_context["input_ids"].squeeze(0),
        "question": tokenized_question["input_ids"].squeeze(0),
        "start_pos": token_start,
        "end_pos": token_end
    }

# Preprocess the entire dataset
processed_data = []
for c, q, a, s in zip(contexts, questions, answers, answer_starts):
    item = preprocess_data(c, q, a, s)
    if item:
        processed_data.append(item)

# Convert to PyTorch Dataset
class SquadDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["context"], item["question"], item["start_pos"], item["end_pos"]

dataset = SquadDataset(processed_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Step 3: Define the RNN-based model
class RNNQuestionAnsweringModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super(RNNQuestionAnsweringModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN for context and question
        self.context_rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.question_rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Linear layers for start and end positions
        self.start_linear = nn.Linear(hidden_dim * 2, 1)  # Bidirectional RNN has 2x hidden_dim
        self.end_linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, context, question):
        # Embed context and question
        context_embedded = self.embedding(context)  # [batch_size, max_context_len, embedding_dim]
        question_embedded = self.embedding(question)  # [batch_size, max_question_len, embedding_dim]
        
        # Pass through RNNs
        context_output, _ = self.context_rnn(context_embedded)  # [batch_size, max_context_len, hidden_dim*2]
        question_output, _ = self.question_rnn(question_embedded)  # [batch_size, max_question_len, hidden_dim*2]
        
        # Use the final question representation (last hidden state)
        question_rep = question_output[:, -1, :]  # [batch_size, hidden_dim*2]
        question_rep = question_rep.unsqueeze(1).repeat(1, context_output.size(1), 1)  # [batch_size, max_context_len, hidden_dim*2]
        
        # Concatenate question representation with context
        merged = torch.cat([context_output, question_rep], dim=-1)  # [batch_size, max_context_len, hidden_dim*4]
        
        # Predict start and end positions
        start_logits = self.start_linear(merged).squeeze(-1)  # [batch_size, max_context_len]
        end_logits = self.end_linear(merged).squeeze(-1)  # [batch_size, max_context_len]
        
        return start_logits, end_logits

# Initialize the model
vocab_size = tokenizer.vocab_size
model = RNNQuestionAnsweringModel(vocab_size)

# Step 4: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Step 5: Train the model
def train_model(model, dataloader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            context, question, start_pos, end_pos = batch
            
            # Move tensors to device
            context = context.to(torch.long)
            question = question.to(torch.long)
            start_pos = start_pos.to(torch.long)
            end_pos = end_pos.to(torch.long)
            
            # Forward pass
            optimizer.zero_grad()
            start_logits, end_logits = model(context, question)
            
            # Compute loss
            start_loss = criterion(start_logits, start_pos)
            end_loss = criterion(end_logits, end_pos)
            loss = (start_loss + end_loss) / 2
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# train_model(model, dataloader)

# # Step 6: Test the model
# def predict_answer(model, context, question):
#     model.eval()
#     with torch.no_grad():
#         tokenized_context = tokenizer(context, truncation=True, padding="max_length", max_length=300, return_tensors="pt")
#         tokenized_question = tokenizer(question, truncation=True, padding="max_length", max_length=50, return_tensors="pt")
        
#         context_ids = tokenized_context["input_ids"]
#         question_ids = tokenized_question["input_ids"]
        
#         start_logits, end_logits = model(context_ids, question_ids)
#         start_idx = torch.argmax(start_logits, dim=-1).item()
#         end_idx = torch.argmax(end_logits, dim=-1).item()
        
#         tokens = tokenizer.convert_ids_to_tokens(context_ids[0][start_idx:end_idx + 1])
#         return tokenizer.convert_tokens_to_string(tokens)

# context = "SQuAD is a dataset for question answering."
# question = "What is SQuAD?"
# print(predict_answer(model, context, question))