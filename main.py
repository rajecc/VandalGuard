import asyncio
import logging
import sys
import cv2
import numpy as np
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from ultralytics import YOLO
from datetime import datetime, timedelta
from config import Config, load_config

config: Config = load_config('.env')

TOKEN = config.tg_bot.token
video_source = # Для тестирования с видеофайлом
target_user_id = # Замените на ID пользователя, которому будут отправляться уведомления
SKIP_FRAMES = 3  # Обрабатывать каждый SKIP_FRAMES кадр
last_vandalism_time = None
COOLDOWN_MINUTES = 3  # Минимальное время между сообщениями о вандализме
vandalism_detection_lock = asyncio.Lock()


dp = Dispatcher()
model = YOLO("VandalGuardModel.pt") # Загрузка модели один раз


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:

    await message.answer("Добро пожаловать!\nЭтот бот будет отправлять уведомления о случаях вандализма, обнаруженных нашей системой при помощи камеры в лифте вашего подъезда.")


async def process_frame(frame: np.ndarray, bot: Bot, target_user_id: int):
    global last_vandalism_time
    current_time = datetime.now()
    try:
        # Выполнение предсказания
        prediction_results = model.predict(source=frame, verbose=False)

        # Результат predict - это список объектов Results
        result = prediction_results[0]

        # В задаче классификации у объекта Results есть атрибут 'probs'
        if result.probs is not None:
            probs = result.probs  # Получаем объект с вероятностями классов

            # Получаем индекс класса с наибольшей вероятностью
            predicted_class_index = probs.top1
            # Получаем уверенность (вероятность) для этого класса
            predicted_confidence = probs.top1conf.item() # .item() для получения числа из тензора

            # Получаем имя предсказанного класса (в нашем случае '0' или '1')
            # Имена классов хранятся в словаре names модели
            class_names = result.names
            predicted_class_name = class_names[predicted_class_index]

            print(f"Предсказанный класс: {predicted_class_name}")
            print(f"Уверенность: {predicted_confidence:.4f}")

            # Интерпретация результата
            if predicted_class_name == '1':
                async with vandalism_detection_lock:
                    if last_vandalism_time is None or (current_time - last_vandalism_time) >= timedelta(minutes=COOLDOWN_MINUTES):
                        await bot.send_message(target_user_id, f"🚨🚨🚨 Обнаружен акт вандализма!!!!! ({current_time.strftime('%H:%M:%S')}) 🚨🚨🚨")
                        last_vandalism_time = current_time
            elif predicted_class_name == '0':
                print("Результат: Вандализм не обнаружен.")
            else:
                print("Результат: Неизвестный класс.") # На всякий случай
        else:
            print("Нет предсказаний на кадре.")

    except Exception as e:
        print(f"Произошла ошибка во время обработки кадра: {e}")


async def video_detection_loop(video_source, bot: Bot, target_user_id: int, skip_frames: int):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть видеопоток или файл: {video_source}")
        return

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Конец видеопотока или ошибка при чтении кадра.")
                break

            frame_count += 1
            if frame_count % skip_frames == 0:
                # Запускаем обработку кадра асинхронно, чтобы не блокировать основной цикл
                asyncio.create_task(process_frame(frame, bot, target_user_id))

            # Небольшая задержка для имитации реального времени (можно настроить)
            await asyncio.sleep(0.03) # Меньшая задержка, так как обрабатывается не каждый кадр

    finally:
        cap.release()
        print("Видеопоток остановлен.")


@dp.message()
async def echo_handler(message: Message) -> None:
    """
    Handler will forward receive a message back to the sender

    By default, message handler will handle all message types (like a text, photo, sticker etc.)
    """
    try:
        await message.send_copy(chat_id=message.chat.id)
    except TypeError:
        await message.answer("Nice try!")


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    asyncio.create_task(video_detection_loop(video_source, bot, target_user_id, SKIP_FRAMES))

    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
