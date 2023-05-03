import logging
from datetime import datetime  
import time 


def create_logger(script_name : str):
    # 로그 생성
    logger = logging.getLogger()
    # 로그의 출력 기준 설정
    logger.setLevel(logging.INFO)
    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    timestamp = time.time()
    date_time = datetime.fromtimestamp(timestamp)
    str_time = date_time.strftime("%m%d_%H%M")
    # log를 파일에 출력
    file_handler = logging.FileHandler(f'./log/{script_name}_{str_time}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger