from utils import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default='./sample.txt')
    parser.add_argument("--numQuestion", type=int, default=10)
    args = parser.parse_args()
    model, tokenizer = load_Summarization_model_tokenizer()

    text_list = doc2Chunk(args.file_name)
    for text in text_list:
        summarized_text = summarizer(text, model=model, tokenizer=tokenizer)
        keywordList = keywordExtraction(summarized_text)
        print(f'summarized_text: {summarized_text}')
        print(f'keywords: {keywordList}')
    

