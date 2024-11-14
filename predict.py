from cog import BasePredictor, Input
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.model_name = "facebook/bart-large-cnn"  # Use BART model for summarization
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        torch.set_num_threads(1) # https://github.com/pytorch/pytorch/issues/36191

    def predict(self, content: str = Input(description="Content of the story"),
                      title: str = Input(description="Title of the story", default="")) -> str:
        text_to_summarize = content + " " + title

        summary_threshold = 1000
        text_to_summarize = text_to_summarize[:summary_threshold]

        input_ids = self.tokenizer(
            text_to_summarize,
            return_tensors="pt",  
            padding=True,
            truncation=True,
            max_length=1024  
        )["input_ids"]

        min_length = int(len(text_to_summarize.split()) * 0.2)  
        max_length = len(text_to_summarize.split())  

        summary_ids = self.model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=2,  
            num_beams=4 
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
