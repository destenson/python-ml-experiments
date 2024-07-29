
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class HierarchicalBERTModel(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(HierarchicalBERTModel, self).__init__()
        self.sentence_encoder = BertModel.from_pretrained(bert_model_name)
        self.document_encoder = torch.nn.GRU(input_size=768, hidden_size=768, batch_first=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    def forward(self, document):
        # Tokenize and encode each sentence
        sentence_embeddings = []
        for sentence in document:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            outputs = self.sentence_encoder(**inputs)
            sentence_embeddings.append(outputs.last_hidden_state[:, 0, :])  # [CLS] token representation
        
        sentence_embeddings = torch.stack(sentence_embeddings).squeeze(1)
        
        # Encode the sequence of sentence embeddings
        document_embedding, _ = self.document_encoder(sentence_embeddings.unsqueeze(0))
        
        return document_embedding

class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, max_span):
        super(AdaptiveAttention, self).__init__()
        self.hidden_size = hidden_size
        self.max_span = max_span
        self.span_param = nn.Parameter(torch.ones(1, hidden_size))

    def forward(self, query, key, value):
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        
        # Apply adaptive span
        span_mask = torch.sigmoid(self.span_param)
        attn_scores = attn_scores * span_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output

class AdaptiveBERTModel(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', max_span=512):
        super(AdaptiveBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.adaptive_attention = AdaptiveAttention(hidden_size=768, max_span=max_span)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Apply adaptive attention to the last hidden state
        adaptive_output = self.adaptive_attention(last_hidden_state, last_hidden_state, last_hidden_state)
        
        return adaptive_output


class HierarchicalAdaptiveBERTModel(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', max_span=512):
        super(HierarchicalAdaptiveBERTModel, self).__init__()
        self.sentence_encoder = BertModel.from_pretrained(bert_model_name)
        self.document_encoder = torch.nn.GRU(input_size=768, hidden_size=768, batch_first=True)
        self.adaptive_attention = AdaptiveAttention(hidden_size=768, max_span=max_span)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    def forward(self, document):
        sentence_embeddings = []
        for sentence in document:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            outputs = self.sentence_encoder(**inputs)
            sentence_embeddings.append(outputs.last_hidden_state[:, 0, :])
        
        sentence_embeddings = torch.stack(sentence_embeddings).squeeze(1)
        document_embedding, _ = self.document_encoder(sentence_embeddings.unsqueeze(0))
        
        adaptive_output = self.adaptive_attention(document_embedding, document_embedding, document_embedding)
        
        return adaptive_output

# if __name__ == '__main__':
    
# EOF
