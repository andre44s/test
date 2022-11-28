import boto3
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from math import ceil
from easyocr import Reader
import string
import streamlit as st
import logging
import collections

def setup_s3_client():
    client = boto3.client('s3',
                    aws_access_key_id = st.secrets["aws_access_key_id"],
                    aws_secret_access_key = st.secrets["aws_secret_access_key"]
                    )
    print("client established")

    return client


def get_csv_from_s3(bucket, key):
    print("bucket : " + bucket + " key : " + key)
    client = setup_s3_client()
    csv_obj = client.get_object(Bucket=bucket, Key=key)
    # print(csv_obj, "bucket response----")
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))
    return df


def get_embedding_from_s3(bucket, key):
    print("bucket : " + bucket + " key : " + key)
    data = BytesIO()
    client = setup_s3_client()
    client.download_fileobj(bucket, key, data)
    data.seek(0)
    embeddings = np.load(data)
    return embeddings


def upload_csv_to_s3(bucket, csv_file, key):
    client = setup_s3_client()
    try:
        with StringIO() as csv_buffer:
            csv_file.to_csv(csv_buffer, index=False)
            print("bucket : " + bucket + " key : " + key)
            response = client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
            # print(response)
    except Exception as e:
        print(e)
        pass


async def url2binary(url, session):
    tries = 5

    for i in range(tries):
        try:
            response = await session.get(url)
            return response
        except:
            continue

    return asyncio.sleep(1)  # returning none if process error


# batch process for async function calling
def get_tasks(session, files, start_index, end_index):
    tasks = []
    for i in files[start_index:end_index]:
        tasks.append(url2binary(i, session))

    return tasks


# main async function + batch calling process
async def get_img(files, batch_count):
    print('Start Downloading Image Files')
    img_files = []
    file_len = len(files)
    async with aiohttp.ClientSession() as session:
        for i in range(ceil(file_len / batch_count)):
            if (i + 1) * batch_count < file_len:
                tasks = get_tasks(session, files, i * batch_count, (i + 1) * batch_count)
                # print(f'Downloading Image Files : {i*batch_count} - {(i+1)*batch_count}')
            else:
                tasks = get_tasks(session, files, i * batch_count, file_len)
                # print(f'Downloading Image Files : {i*batch_count} - {file_len}')

            responses = await asyncio.gather(*tasks)
            for response in responses:
                if type(response) == aiohttp.ClientResponse:
                    img_files.append(await response.read())
                else:
                    img_files.append(await response)

    print(f'Finished Downloading Image Files : {file_len} Images')

    return img_files


def remove_punctuations(text):
    for punctuation in string.punctuation:
        try:
            text = text.replace(punctuation, '')
        except:
            print(text)
            # text = 'invalid_subject'

    return text


def get_reader():
    reader = Reader(['en'], gpu=False)

    return reader


def extract_text(img, reader):
    st.text("Proc")
    result = ['Nothing']
    g, t, c = ['Nothing'], ['Nothing'], ['Nothing']
    try:
        result = reader.readtext(img, detail=1)
        if result != []:
            g, t, c = extract_bb_text_confidence(result)
        else:
            raise Exception("empty result")
    except Exception as e:
        st.text(e)
        pass

    return g, t, c


def split_text_get_image_name(x):
    return x.split('/')[-1]


def list_to_string(x):
    return ' '.join(x)


def bounding_box_sorting(boxes):
    num_boxes = len(boxes)
    # sort from top to bottom and left to right
    sorted_boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    # print('::::::::::::::::::::::::::testing')
    # check if the next neighgour box x coordinates is greater then the current box x coordinates if not swap them.
    # repeat the swaping process to a threshold iteration and also select the threshold
    threshold_value_y = 20
    for j in range(5):
        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < threshold_value_y and (
                    _boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp

    return _boxes


def extract_bb_text_confidence(geometry_text_confidence):
    my_dict = dict(zip(np.arange(len(geometry_text_confidence)), [i[0] for i in geometry_text_confidence]))
    sorted_geometry = [list(my_dict.keys())[list(my_dict.values()).index(i)] for i in
                       bounding_box_sorting([i[0] for i in geometry_text_confidence])]
    result = [geometry_text_confidence[i] for i in sorted_geometry]
    g, t, c = [i[0] for i in result], [i[1] for i in result], [i[2] for i in result]

    return [g, t, c]


def text_to_embeddings(embedder1, embedder2, text):
    text_embedding1 = embedder1.encode(text, convert_to_tensor=False, normalize_embeddings=True)
    text_embedding2 = embedder2.encode(text, convert_to_tensor=False, normalize_embeddings=True)
    text_embedding = np.hstack((text_embedding1, text_embedding2))

    return text_embedding


# logging class
class TailLogHandler(logging.Handler):

    def __init__(self, log_queue):
        logging.Handler.__init__(self)
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.append(self.format(record))


# loggin class
class TailLogger(object):

    def __init__(self, maxlen):
        self._log_queue = collections.deque(maxlen=maxlen)
        self._log_handler = TailLogHandler(self._log_queue)

    def contents(self):
        return '\n'.join(self._log_queue)

    @property
    def log_handler(self):
        return self._log_handler


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')