import logging
import os

from datetime import datetime

FILENAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
FILEPATH = os.path.join(os.getcwd(), "logs")
os.makedirs(FILEPATH, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(FILEPATH, FILENAME),
    format="[%(asctime)s] - %(pathname)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
