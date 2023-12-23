# Brand ML Hackathon
 
**Описание решения**:

В датасете, который предоставили организаторы, я оставил только комменты, непосредственным родителем которых являются посты (т.е. удалил комменты, которые отвечают на другие комменты). Другой предобработки данных не делал. Затем, так как в датасете отсутствует разметка, я самостоятельно сгенерировал ее с помощью Yandex GPT. Далее взял модель FRED-T5 от Сбера с huggingface ("ai-forever/FRED-T5-1.7B"), квантизовал ее в 4 бита (чтобы она поместилась на GPU в Google Colab) и дообучил с помощью метода LoRA.

Для того, чтобы генерировать разные саммари в зависимости от того, хотим ли мы суммаризовать все комменты, комменты непосредственно относящиеся к посту или комменты косвенно относящиеся к посту, я использую Bert-like модель с huggingface ('cointegrated/rubert-tiny2'). Для каждого поста я пробегаюсь по всем комментам и оставляю только те, чья косинусная близость попадает в определенный диапазон: для всех комментов (косинусное расстояние больше 0.3), для комментов непосредственно относящихся к посту (косинусное расстояние больше 0.55), для комментов косвенно относящихся к посту (косинусное расстояние от 0.3 до 0.55). Для отобранных комментов генерю саммари зафайнтюненой моделью FRED-T5, о которой писал выше.

**Запуск**:

1. Добавьте файл с данными в папку с файлом solution.py.
2. Установите зависимости из файла dependencies.txt
3. Запускайте в консоли с помощью команд:


   python3 ./solution.py all_comments "./dataset.jsonl" "./result.jsonl"
   
   python3 ./solution.py post_comments "./dataset.jsonl" "./result.jsonl"
   
   python3 ./solution.py topic_comments "./dataset.jsonl" "./result.jsonl"

**Примечание**:

Я не уверен на 100%, что все запустится :)

Все отлично работает на Google Colab, но у меня на Windows запустить не получилось из-за несовместимости библиотеки bitsandbytes с CUDA драйверами под Windows. По идее, если у вас сервер на Ubuntu, то работать должно. 




 
