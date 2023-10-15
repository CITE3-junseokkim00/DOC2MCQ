import numpy as np
import random
import torch
from transformers import T5TokenizerFast
from transformers.models.t5 import T5ForConditionalGeneration
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def doc2Chunk(file_path):
    raw_text=''
    if '.pdf' in file_path:
        doc_reader = PdfReader(file_path)
        for i,page in enumerate(doc_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text.strip()
    else:
        with open(file_path, 'r') as f:
            raw_text = f.read()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )

    texts = text_splitter.split_text(raw_text)
    texts = [text.replace('\n',' ') for text in texts]
    return texts



# KEYPHRASE EXTRACTION TASK
sentence_model = SentenceTransformer(model_name_or_path='paraphrase-multilingual-mpnet-base-v2')
kw_model = KeyBERT(model=sentence_model)
okt = Okt()

def keywordExtraction(document):
    keywords = kw_model.extract_keywords(docs=document,keyphrase_ngram_range=(1,1),stop_words=None,top_n=5)
    keywordset = []
    keywords = [keyword[0] for keyword in keywords]
    for i in keywords:
        pos = okt.pos(i)
        for word in pos:
            if word[1] == 'Noun' and word[0] not in keywordset:
                keywordset.append(word[0])
    return keywordset


# SUMMARIZATION TASK
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_Summarization_model_tokenizer():
    # model = T5ForConditionalGeneration.from_pretrained('./koT5_summary/model').to(device)
    model = T5ForConditionalGeneration.from_pretrained('./koT5_summary/model')
    tokenizer = T5TokenizerFast.from_pretrained('./koT5_summary/tokenizer')
    return model, tokenizer

def summarizer(text,model,tokenizer):
    text = text.strip().replace("\n"," ")
    text = "Summarize: " + text
    # max_len = 512
    max_len = 1024
    # encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors='pt').to(device)
    encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors='pt')
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=3,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          min_length=150,
                          max_length=300)
    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]

    return summary.strip()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)