#﻿## Brand ML Hackathon
 
##Описание решения:

В датасете, который предоставили организаторы, я оставил только комменты, непосредственным родителем которых являются посты (т.е. удалил комменты, которые отвечают нв другие комменты). Другой предобработки данных не делал. Затем, так как в датасете отсутствует разметка, я самостоятельно сгенерировал ее с помощью Yandex GPT. Далее взял модель FRED-T5 от Сбера с huggingface ("ai-forever/FRED-T5-1.7B"), квантизовал ее в 4 бита (чтобы она поместилась на GPU в Google Colab) и дообучил с помощью метода LoRA.
Для того, чтобы генерировать разные саммари в зависимости от того, хотим ли мы суммаризовать все комменты, комменты непосредственно относящиеся к посту или комменты косвенно относящиеся к посту, я использую Bert-like модель с huggingface ('cointegrated/rubert-tiny2'). Для каждого поста я пробегаюсь по всем комментам и оставляю только те, чья косинусная близость попадает в определенный диапазон: для всех комментов (косинусное расстояние больше 0.3), для комментов непосредственно относящихся к посту (косинусное расстояние больше 0.55), для комментов косвенно относящихся к посту (косинусное расстояние от 0.3 до 0.55). Для отобранных комментов генерю саммари зафайнтюненой моделью FRED-T5, о которой писал выше.

 
