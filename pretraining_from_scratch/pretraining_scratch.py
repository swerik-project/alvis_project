from transformers import PreTrainedTokenizerFast
from transformers import BertConfig as TransformersBertConfig
from typing import  cast
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
import pickle
import preprocessing
import argparse
import bert_layers as bert_layers_module
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader



class BertConfig(TransformersBertConfig):

    def __init__(
        self,
        alibi_starting_size: int = 512,
        attention_probs_dropout_prob: float = 0.0,
        **kwargs,
    ):
        """Configuration class for MosaicBert.

        Args:
            alibi_starting_size (int): Use `alibi_starting_size` to determine how large of an alibi tensor to
                create when initializing the model. You should be able to ignore this parameter in most cases.
                Defaults to 512.
            attention_probs_dropout_prob (float): By default, turn off attention dropout in Mosaic BERT
                (otherwise, Flash Attention will be off by default). Defaults to 0.0.
        """
        super().__init__(
            attention_probs_dropout_prob=attention_probs_dropout_prob, **kwargs)
        self.alibi_starting_size = alibi_starting_size



def tokenize(element,tokenizer,context_length):
    outputs = tokenizer(
        element["texte"],
        truncation=False,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    eos_token_id = tokenizer.sep_token_id
    print(eos_token_id)
    concatenated_sequence = []

    # Append each tokenized input with the eos_token_id and flatten into a single list
    for ids in outputs["input_ids"]:
        concatenated_sequence.extend(ids + [eos_token_id])

    # Remove the last eos_token_id if it's at the end of the sequence
    if concatenated_sequence[-1] == eos_token_id:
        concatenated_sequence.pop()

    # Chunk the concatenated sequence into segments of context_length
    input_batch = []
    for i in range(0, len(concatenated_sequence), context_length):
        chunk = concatenated_sequence[i:i + context_length]
        if len(chunk) == context_length:
            input_batch.append(chunk)

    # Return the chunked sequences
    return {"input_ids": input_batch}


def tokenizer(file):
    return PreTrainedTokenizerFast(
    tokenizer_file=file,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
    )

def main(args):
    print("loading tokenizer...")
    wrapped_tokenizer = tokenizer(args.tokenizer_file)

    print("Loading model...")
    with open("examples/examples/benchmarks/bert/yamls/main/mosaic-bert-base-uncased.yaml") as f:
        yaml_cfg = om.load(f)
    cfg = cast(DictConfig, yaml_cfg) #configuration for mosaicBert (from Mosaicml github)

    pretrained_model_name = args.model_checkpoint
    model_config=cfg.model.get('model_config', None)
    config = BertConfig.from_pretrained(
            pretrained_model_name, **model_config)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8) #size of vocabulary mutlple of 8
    
    model = bert_layers_module.BertForMaskedLM(config)
        # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    model.resize_token_embeddings(config.vocab_size)

    print("Loading Data collector..")
    mlm_probability = args.mlm
    collate_fn = DataCollatorForLanguageModeling(
            tokenizer=wrapped_tokenizer,
            mlm=mlm_probability is not None,
            mlm_probability=mlm_probability)
    print("Loading Dataset ...")
    
    #data_files={"train":"swerick_data_random_train.pkl","test":"swerick_data_random_test.pkl","valid":"swerick_data_random_valid.pkl"}
    #swerick_dataset = load_dataset("pandas",data_files=data_files)
    
    #tokenized_datasets = swerick_dataset.map(
    #lambda batch: tokenize(batch, wrapped_tokenizer, args.context_length), batched=True, remove_columns=swerick_dataset["train"].column_names
#)
    
    with open("from_scratc_dataset.pkl","rb") as f:
        tokenized_datasets = pickle.load(f)


    
    train_dataloader = DataLoader(tokenized_datasets["train"],collate_fn=collate_fn,batch_size=cfg.global_train_batch_size,num_workers=cfg.train_loader.num_workers)
    test_dataloader = DataLoader(tokenized_datasets["test"],batch_size=cfg.global_train_batch_size,num_workers=cfg.train_loader.num_workers)

    
    logging_steps = len(tokenized_datasets["train"]) // cfg.global_train_batch_size

    trainer = preprocessing.create_trainer(model,args.name,cfg.global_train_batch_size,logging_steps,learning_rate=cfg.optimizer.lr,decay=cfg.optimizer.weight_decay,train_dataset=tokenized_datasets["train"],eval_dataset=tokenized_datasets["test"],data_collator=collate_fn,tokenizer=wrapped_tokenizer,num_epochs=args.epochs)
    trainer.args.lr_scheduler_type = args.lr_type
    if args.lr_type =='linear':
        trainer.args.warmup_ratio=0.01 #same as BERT model
    trainer.train(resume_from_checkpoint= args.trainer_checkpoint)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer_file", type=str, default="tokenizer_swerick.json", help="Save location for the tokenizer")
    parser.add_argument("--model_checkpoint", type=str, default="KBLab/bert-base-swedish-cased", help="Save location for checkpoint of the trainer")
    parser.add_argument("--trainer_checkpoint", type=str, default=None, help="Save location for checkpoint of the trainer")
    parser.add_argument("--name", type=str, default="scratch_pretraining", help="repository name")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--lr_type", type=int, default='linear')
    parser.add_argument("--mlm", type=int, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    main(args)