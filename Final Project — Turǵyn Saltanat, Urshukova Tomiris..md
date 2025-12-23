
# Проект: Классификация видеоигр

## Авторы

* Тұрғын Салтанат
* Уршукова Томирис

## Описание проекта

В рамках финального проекта проведён анализ датасета видеоигр с использованием **линейной и логистической регрессии с нуля**, а также **Decision Tree**.
Проект демонстрирует:

* Реализацию градиентного спуска для линейной и логистической регрессии
* Использование mini-batch градиентного спуска
* Визуализацию данных (scatter plot, регрессионная линия, доверительный интервал)
* Сравнение моделей по метрикам: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC AUC
* Эксперименты с различными learning rate, epoch и batch size
* Интерактивный интерфейс с виджетами для выбора модели и параметров обучения

## Датасет

Использован датасет с Kaggle:
[IMDb Video Games Dataset] https://www.kaggle.com/datasets/lorentzyeung/imdb-video-games-dataset

Количество строк: **14682**

Основные используемые колонки:

* **User Rating** — независимая переменная (X)
* **Popularity** — зависимая переменная (y)

## Структура проекта

* `Final Project.ipynb` — основная реализация моделей, графики и интерфейс
* `README.md` — описание проекта и инструкция по запуску
* `requirements.txt` — необходимые библиотеки

## Запуск проекта в Google Colab

1. Откройте Notebook в Colab: https://colab.research.google.com/drive/1AasfEoZ9XuyUe95TFJgGBxnr40dwAqD_?usp=sharing
2. При необходимости установите зависимости:

```python
!pip install numpy pandas matplotlib seaborn scikit-learn ipywidgets scipy
```

3. Используйте интерактивные виджеты для выбора модели и параметров обучения.

---

## Используемые модели

* **Линейная регрессия** — предсказание численной популярности
* **Логистическая регрессия (с нуля)** — бинарная классификация
* **Decision Tree** — сравнение с нелинейной моделью

## Метрики оценки

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC AUC

## Эксперименты

* Влияние **learning rate**
* Влияние **epoch**
* Mini-batch градиентный спуск
* Сравнение трёх моделей



---

### ✅ Линейная регрессия (inline LaTeX)

Функция предсказания:
`$\hat{y} = wx + b$`

Функция потерь:
`$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - (wx_i + b))^2$`

Градиент по весу:
`$\frac{\partial L}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n} x_i (y_i - (wx_i + b))$`

Градиент по смещению:
`$\frac{\partial L}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n} (y_i - (wx_i + b))$`

---

### ✅ Логистическая регрессия (inline LaTeX)

Сигмоида:
`$\sigma(z) = \frac{1}{1 + e^{-z}}$`

Функция предсказания:
`$\hat{y} = \sigma(wx + b)$`

Log-loss:
`$L = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]$`

Градиент по весу:
`$\frac{\partial L}{\partial w} = \frac{1}{n}X^T(\hat{y} - y)$`

Градиент по смещению:
`$\frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$`

---

# requirements.txt

```
numpy
pandas
matplotlib
seaborn
scikit-learn
ipywidgets
scipy
```


