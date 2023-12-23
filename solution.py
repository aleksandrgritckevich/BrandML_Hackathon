import src.predict
import sys
import json
import gc
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, TaskType

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    type_comments = sys.argv[1]
    data_path = sys.argv[2]
    json_path = sys.argv[3]
    if type_comments == "all_comments":
        min_threshold, max_threshold = 0.3, 1
    elif type_comments == "post_comments":
        min_threshold, max_threshold = 0.55, 1
    elif type_comments == "topic_comments":
        min_threshold, max_threshold = 0.3, 0.55
    else:
        raise ValueError('Значение аргумента должно быть одним из следующих: "all_comments", "post_comments", "topic_comments"')
    # Загружаем и предобрабатываем данные. Оставляем только комменты к посту, т.е. удаляем комменты к комментам.

    posts_comments = src.predict.process_data(data_path)


    # Используем маленькую Bert-like модель с huggingface, чтобы оценить релевантность каждого коммента к основному посту и
    # отсеять комменты, которые не проходят threshold. Для модели, суммаризирующей все комменты я использую threshold 0.3,
    # чтобы убрать совсем нерелевантные комментарии. Для модели, которая суммаризирует только комменты,
    # непосделственно относящиеся к посту ставлю threshold равный 0.55. Для модели суммаризирующей комменты, косвенно
    # относящиеся к посту, ставлю threshold минимальный 0.3 и максимальный 0.55.

    model = SentenceTransformer('cointegrated/rubert-tiny2').to(device)
    features = src.predict.filter_comments(posts_comments, model, min_threshold, max_threshold)

    # Удаляем модель, чтобы освободить память GPU.

    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Предсказываем суммаризации с помощью большой модели FRED-T5, которую я дообучил на разметке, которую самостоятельно
    # сгенерировал с помощью Yandex GPT. Модель загружаем в квантизированном виде, чтобы она влезла на гпу и дообучаем
    # с помощью метода LoRA, который позволяет не обучать все параметры модели, а только небольщую добавку к весам.

    model_name = "ai-forever/FRED-T5-1.7B"
    adapters_name = "chibeenot/awesome"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )

    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=16, lora_alpha=32,
                             lora_dropout=0.1)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(model, adapters_name, token="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")

    # Предсказываем саммари на нашем датасете.

    prediction, hashes, comments_hashes = src.predict.predict(features,model,tokenizer)

    # Сохраняем все в json.
    to_json = [{"summary": prediction[i], "post_hash": hashes[i], "comments_hash": comments_hashes[i]} for i in
               range(len(prediction))]
    with open(json_path, 'w') as f:
        for d in to_json:
            json.dump(d, f)
            f.write('\n')
