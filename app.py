from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import os
from io import BytesIO

app = Flask(__name__)

# Загрузка модели
try:
    generator_model = tf.keras.models.load_model('generator_model.keras')
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit(1)  # Выход с ошибкой


def generate_image(noise_dim=100):
    noise = np.random.normal(0, 1, (1, noise_dim))
    generated_image = generator_model.predict(noise)
    # Нормализация изображения
    generated_image = np.clip(generated_image, 0, 1)  # Важно!
    img = Image.fromarray((generated_image[0] * 255).astype(np.uint8))
    img = img.convert('RGB') # Преобразование в RGB если необходимо
    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            generated_image = generate_image()
            # Сохранение изображения в памяти
            img_bytes = generated_image.tobytes()
            img_format = 'PNG'
            
            #Возвращаем изображение как base64
            buffer = BytesIO()
            generated_image.save(buffer, format=img_format)
            img_str = 'data:image/' + img_format + ';base64,' + \
                      base64.b64encode(buffer.getvalue()).decode('utf-8')
            return render_template('index.html', img_src=img_str)

        except Exception as e:
            return jsonify({'error': str(e)}), 500  # Возвращаем ошибку

    return render_template('index.html', img_src=None)


if __name__ == '__main__':
    app.run(debug=True)