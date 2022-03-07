from transformers import BertModel
from src.modules import CRF
import torch
from torch import nn

class TripletTagger(nn.Module):
    def __init__(self, bert_hidden_dim, num_binslot=3):
        super(TripletTagger, self).__init__()
        
        self.num_binslot = num_binslot
        self.hidden_dim = bert_hidden_dim
        self.linear = nn.Linear(self.hidden_dim, self.num_binslot)
        self.crf_layer = CRF(self.num_binslot)
        
    def forward(self, inputs, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            prediction: logits of predictions
            crf_loss: loss of crf
        """
        prediction = self.linear(inputs)
        crf_loss = self.crf_layer.loss(prediction, y)
        return prediction, crf_loss
    
    def crf_decode(self, logits):
        """
        crf decode
        logits to labeling (0/1/2 == O/B/I)
        Input:
            logits: (bsz, max_seq_len, num_entity)
        Output:
            pred: (bsz, max_seq_len)
        """
        return torch.argmax(logits, dim=2)

class SlotFillingModel(nn.Module):
    def __init__(self, args):
        super(SlotFillingModel, self).__init__()

        # hyperparameters
        self.num_tags = args.num_tags
        self.key_ratio = args.loss_key_ratio
        self.momentum = args.momentum
        self.key_updated_cnt = 0
        self.key_update_period = args.key_update_period

        # models
        if args.model_name_or_path == "lstm":
            self.query_bert_emb = torch.nn.Embedding(args.vocab_size, args.hidden_size, args.pad_token_id)
            self.query_bert = torch.nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size)
            # self.key_bert_emb = torch.nn.Embedding(args.vocab_size, args.hidden_size, args.pad_token_id)
            # self.key_bert = torch.nn.LSTM(input_size=args.input_size, hidden_size=args.hidden_size)
        elif "bert" in args.model_name_or_path:
            self.query_bert = BertModel.from_pretrained("bert-base-uncased")

        self.key_bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.classifier = nn.Linear(self.query_bert.config.hidden_size, self.num_tags)
        self.crf = CRF(self.num_tags)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

        # freeze parameters of key encoder
        for param in self.key_bert.parameters():
            param.requires_grad = False

    def forward(self, q_inputs, k_inputs=None, ):
        """
        parameters
        ----------
        q_inputs: consist of (tokenized (slot_desc, utterance) pairs, (gold) bio labels)
        k_inputs: consist of (tokenized templates(, tokenized augmented data))
        """

        with torch.no_grad():
            self.update_key_enc()

        query_outputs = self.query_bert(
            q_inputs['input_ids'],
            q_inputs['attention_mask'],
            q_inputs['token_type_ids']
        )
        query_sequence_outputs = query_outputs.last_hidden_state
        query_cls_output = query_sequence_outputs[:, 0, :]
        query_batch_length = query_cls_output.shape[0]

        # first: BIO classification
        outputs = self.dropout(query_sequence_outputs)
        logits = self.classifier(outputs)
        crf_loss = self.crf.loss(logits, q_inputs['labels'])
        cl_loss = None
        
        # second: query-key contrastive learning
        # only acts when training
        if k_inputs is not None:
            k_input_reshape = {}
            key_batch_length, key_per_query, _ = k_inputs['input_ids'].shape
            for k, v in k_inputs.items():
                if len(v.shape) == 3:
                    k_input_reshape[k] = v.reshape(key_batch_length * key_per_query, -1)

            key_outputs = self.key_bert(
                k_input_reshape['input_ids'],
                k_input_reshape['attention_mask'],
                k_input_reshape['token_type_ids'],
            )
            key_cls_output = key_outputs.last_hidden_state[:, 0, :]
            # key input: [pos_tem/pos_aug, neg_tem1, neg_tem2, ...., neg_aug1, neg_aug2, ...]
            key_cls_output = torch.reshape(key_cls_output, (key_batch_length, key_per_query, -1))

            batch_label = torch.tensor([0 for _ in range(query_batch_length)]).cuda()

            query_cls_output.unsqueeze_(1)
            key_cls_output = torch.transpose(key_cls_output, 1, 2) # shape: (num_batch, bert_hidden_size, num_adaption_data)

            query_key_mult = torch.bmm(query_cls_output, key_cls_output).squeeze(1)

            cl_loss = self.ce_loss(query_key_mult, batch_label)
            self.key_updated_cnt += 1

        return crf_loss, logits, cl_loss

    def update_key_enc(self):
        if self.key_updated_cnt % self.key_update_period == 0:
            for q_params, k_params in zip(self.query_bert.parameters(), self.key_bert.parameters()):
                k_params = self.momentum * k_params + (1 - self.momentum) * q_params

        return