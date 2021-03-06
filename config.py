from curses import meta
from dataclasses import dataclass, field
from email.policy import default
from importlib.metadata import metadata
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    num_tags: int = field(
        default=3,
        metadata={"help": "number of tags"}
    )
    dropout_rate: float = field(
        default=0.5,
        metadata={"help": "dropout rate"}
    )
    key_enc_data: str = field(
        default="tem_aug",
        metadata={"help": "designate data for key encoder input (\"tem_only\", \"tem_aug\", \"aug_only\")"}
    )
    use_both_or_not: bool = field(
        default=False,
        metadata={"help": "use template and aug_data both or not, this is only valid for key_enc_data = \"tem_aug\""}
    )
    num_key_enc_data: int = field(
        default=3,
        metadata={"help": "number of data for key encoder input"}
    )
    num_aug: int = field(
        default=1,
        metadata={"help": "number of augmented data when training"}
    )
    loss_key_ratio: float = field(
        default=0.5,
        metadata={"help":"ratio of usage of key enc's output"}
    )
    momentum: float = field(
        default=0.999, 
        metadata={"help": "hyperparameter for update key encoder"}
    )
    key_update_period: int = field(
        default=10,
        metadata={"help": "hyperparameter period for update key encoder"}
    )
    # for LSTM usage
    vocab_size: int = field(
        default=30522,
        metadata={}
    )
    hidden_size: int = field(
        default=768,
        metadata={}
    )
    pad_token_id: int = field(
        default=0
    )
    



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(default="slot_filling", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_path: str = field(
        default="./data",
        metadata={"help": "path of dataset"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    target_domain: str = field(
        default="AddToPlaylist",
        metadata={"help": "target domain"}
    )
    n_samples: int = field(
        default=0,
        metadata={"help": "number of samples for few shot learning"}
    )
    early_stopping_patience: int = field(
        default=30,
        metadata={"help": "patience for early stopping"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

# do not use below
# def get_params():
#     # parse parameters
#     parser = argparse.ArgumentParser(description="Cross-domain SLU with BERT")
#     parser.add_argument("--use_plain", type=bool, default=True, help="if use BERT_NER False elif BERT_Plain True")
#     parser.add_argument("--exp_name", type=str, default="Plain_model", help="Experiment name")
#     parser.add_argument("--logger_filename", type=str, default="cross-domain-slu.log")
#     parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
#     parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")

#     # adaptation parameters
#     parser.add_argument("--epoch", type=int, default=10, help="number of maximum epoch")
#     parser.add_argument("--tgt_dm", type=str, default="", help="target_domain")
#     parser.add_argument("--batch_size", type=int, default=128, help="batch size")
#     parser.add_argument("--num_binslot", type=int, default=3, help="number of binary slot O,B,I")
#     parser.add_argument("--num_slot", type=int, default=72, help="number of slot types")
#     parser.add_argument("--num_domain", type=int, default=7, help="number of domain")
    
#     parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
#     parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")

#     # few shot learning
#     parser.add_argument("--n_samples", type=int, default=0, help="number of samples for few shot learning")

#     # test model
#     parser.add_argument("--model_path", type=str, default="", help="Saved model path")
#     parser.add_argument("--model_type", type=str, default="", help="Saved model type (e.g., coach, ct, rzt)")
#     parser.add_argument("--test_mode", type=str, default="testset", help="Choose mode to test the model (e.g., testset, seen_unseen)")

#     params = parser.parse_args()

#     return params
