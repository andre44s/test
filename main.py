import streamlit as st
import datetime
import pandas as pd
from io import StringIO
import re
import logging
from sentence_transformers import SentenceTransformer, util
import numpy as np
import ast
from joblib import Parallel, delayed
import gc
import utils
from utils import TailLogger
import asyncio

origin_bucket = st.secrets["origin_bucket"]
origin_prefix = st.secrets["origin_prefix"]

destination_bucket = st.secrets["destination_bucket"]
destination_prefix = st.secrets["destination_prefix"]
destination_file = destination_prefix + 'processed_file_id.csv'

client = utils.setup_s3_client()
reader = utils.get_reader()

@st.cache
def read_day(origin_key):
    csv_obj = client.get_object(Bucket=origin_bucket, Key=origin_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    
    return pd.read_csv(StringIO(csv_string))

st.set_page_config(page_title="MSD Data Processing", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("MSD Data Processing")
tab1, tab2 = st.tabs(["MSD QA Count", "MSD Data File Processing"])


with tab1: # === MSD QA COUNT ===
    # two columns for the start date and end date
    col1, col2 = st.columns(2)
    with col1:
        start_date_selector = st.date_input(
            label="Start Date",
            value=datetime.datetime.today() - datetime.timedelta(days=1),
            min_value=datetime.datetime.today() - datetime.timedelta(days=90),
            max_value=datetime.datetime.today() - datetime.timedelta(days=1),
            key='start_date'
            )

    with col2:
        end_date_selector = st.date_input(
            label="End Date",
            value=datetime.datetime.today() - datetime.timedelta(days=1),
            min_value=datetime.datetime.today() - datetime.timedelta(days=90),
            max_value=datetime.datetime.today() - datetime.timedelta(days=1),
            key='end_date'
            )

    qa_proccess_type = st.radio('QA Type', ['Regular Search', 'Text Search'], index=0)

    # error correcting if user missinput between start date and end date
    start_date = min([start_date_selector, end_date_selector])
    end_date = max([start_date_selector, end_date_selector])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button('Process', key='process'):
            with st.spinner('Loading Files'):
                # load file from s3 

                # downloading the data for each day 
                date_range = pd.date_range(start_date, end_date, freq='d')
                read_fail = []
                raw_dataset = pd.DataFrame()
                for date in date_range:
                    if qa_proccess_type == "Regular Search":
                        origin_key = origin_prefix+str(date)[0:10]+'/000.csv'
                        print('Choosing Regular Search')
                    elif qa_proccess_type == "Text Search":
                        origin_key = origin_prefix+str(date)[0:10]+'/text-000.csv'
                        print('Choosing Text Search')
                    else:
                        origin_key = origin_prefix+str(date)[0:10]+'/000.csv'


                    try:
                        df = read_day(origin_key)
                        raw_dataset = raw_dataset.append(df, ignore_index = True)
                        print(f"{origin_key} success")
                    except:
                        st.warning(f"{origin_key} failed")
                        print(f"{origin_key} failed")
                        read_fail.extend(origin_key)

                destination_file = destination_prefix + 'processed_file_id.csv'
                obj = client.get_object(Bucket=destination_bucket, Key=destination_file)
                previous_file = pd.read_csv(obj['Body'])  # 'Body' is a key word

                all_file_ids = list(previous_file['file_id'].values)
                last_file_id = all_file_ids[-1]
                curr_id = last_file_id+1

                # logging initialization
                logger = logging.getLogger("__process__") # creating logging variable
                logger.setLevel(logging.DEBUG) # set the minimun level of loggin to DEBUG
                formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s") # logging format that will appear in the log file
                tail = TailLogger(10000) # amount of log that is saved
                log_handler = tail.log_handler # variable log handler
                log_handler.setFormatter(formatter) # set formatter to handler
                logger.addHandler(log_handler) # adding file handler to logger

                try:
                    logger.info("QA User Billing Start")
                    dataset = raw_dataset.copy()

                    #data mapping and feedback labeling
                    dataset['created_date'] = pd.to_datetime(dataset['created_date'])
                    dataset['feedback_tags1'] = dataset['extras.tags'].astype(str).apply(utils.remove_punctuations)
                    dataset['extras.subject'] = dataset['extras.subject'].astype(str).apply(utils.remove_punctuations)
                    # dataset['extras.image_type'] = dataset['extras.image_type'].astype(str).apply(utils.remove_punctuations)
                    feedback_label = {'different' : 0, '25 Similar' : 1, '50 Similar' : 2, '75 Similar' : 3, '100 Similar' : 4, 'exact' : 5}
                    dataset['feedback_tags'] = dataset['feedback_tags1'].map(feedback_label)

                    undefined_user_list = ["VUEFIND_undefined", "VUEFIND_null"]
                    undefined_user = dataset[dataset['qa_user_id'].isin(undefined_user_list)]
                    dataset = dataset[~dataset['qa_user_id'].isin(undefined_user_list)]
                    dataset = pd.concat([dataset, undefined_user])

                    # copy tagging data from index = -1 to all index
                    dataset['new_index'] = dataset['index']
                    dataset['new_index'][dataset['index'] == -2] = 10

                    if qa_proccess_type == "Regular Search":
                        tag_columns = ['extras.flagged', 'extras.sample_type', 'extras.image_type',
                                    'extras.picture_taken', 'extras.subject', 'extras.section',
                                    'extras.chapter', 'extras.topic', 'extras.not_found_in_index']
                    elif qa_proccess_type == "Text Search":
                        tag_columns = ['extras.flagged', 'extras.image_type',
                                    'extras.picture_taken', 'extras.subject', 'extras.section',
                                    'extras.chapter', 'extras.topic']

                    dataset.sort_values(by = ['request_id', 'qa_user_id', 'new_index'], inplace = True)
                    dataset[tag_columns] = dataset.groupby(['request_id', 'qa_user_id'])[tag_columns].fillna(method='ffill')

                    dataset = dataset.drop('new_index', axis=1)

                    # keeping later date data for same request_id and index for same qa user
                    dataset1 = dataset.groupby(['request_id', 'qa_user_id', 'index'], group_keys=False, as_index=False).apply(lambda x: x.loc[x.created_date.idxmax()])

                    total_qa_users = dataset1['qa_user_id'].nunique()
                    unique_request_ids = dataset1['request_id'].nunique()
                    index_wise_requests = dataset1[['index', 'request_id']].groupby('index').nunique()

                    # Invalid data
                    invalid = ['Not a relevant subject question', 'Invalid Question']
                    words_re = re.compile("|".join(invalid))
                    if qa_proccess_type == "Regular Search":
                        dataset1['is_valid'] = dataset1['extras.image_type'].apply(lambda x: False if words_re.search(str(x)) else True)
                    elif qa_proccess_type == "Text Search":
                        dataset1['is_valid'] = dataset1['extras.completion'].apply(lambda x: False if words_re.search(str(x)) else True)

                    invalid_requests = dataset1[dataset1['is_valid']==False]

                    # Exact first position
                    combined_invalid = invalid_requests.groupby('qa_user_id')['request_id'].apply(set).reset_index(name='request_list_invalid')
                    dataset1 = pd.merge(dataset1, combined_invalid, how='left', on=['qa_user_id'])

                    in_combined_invalid = []
                    for i in dataset1.index:
                        in_combined_invalid += [True] if (pd.notnull(dataset1.loc[i, 'request_list_invalid']) and dataset1['request_id'][i] in dataset1['request_list_invalid'][i]) else [False]

                    dataset1['in_combined_invalid'] = in_combined_invalid

                    exact_first_position = dataset1[(dataset1['index']==0) & (dataset1['feedback_tags']==5) & (dataset1['in_combined_invalid']==False)]

                    # tagged -2 index
                    exact_invalid = pd.concat([exact_first_position, invalid_requests], axis=0)
                    combined_exact_invalid = exact_invalid.groupby('qa_user_id')['request_id'].apply(set).reset_index(name='request_exact_invalid')
                    dataset1 = pd.merge(dataset1, combined_exact_invalid, how='left', on=['qa_user_id'])

                    in_combined_exact_invalid = []
                    for i in dataset1.index:
                        in_combined_exact_invalid += [True] if (pd.notnull(dataset1.loc[i, 'request_exact_invalid']) and dataset1['request_id'][i] in dataset1['request_exact_invalid'][i]) else [False]

                    dataset1['in_combined_exact_invalid'] = in_combined_exact_invalid

                    feedback_list = dataset1[dataset1['index']>=0].groupby(['request_id', 'qa_user_id'])['feedback_tags'].apply(list).reset_index(name='feedback_list')
                    neg2_requests = dataset1[(dataset1['index']==-2) & (dataset1['feedback_tags'] >= 4) & (dataset1['in_combined_exact_invalid'] == False)]
                    neg2_requests = neg2_requests.merge(feedback_list, how = 'inner', on = ['request_id', 'qa_user_id'])
                    neg2_requests['has_positive'] = neg2_requests['feedback_list'].apply(lambda x : True if any(i >=4 for i in x) else False)
                    neg2_requests = neg2_requests[neg2_requests['has_positive']==False]

                    # tagged all 10
                    combined3 = pd.concat([exact_first_position, invalid_requests, neg2_requests], axis=0)
                    combined3_req_id = combined3.groupby('qa_user_id')['request_id'].apply(set).reset_index(name='request_list_all3')
                    dataset1 = pd.merge(dataset1, combined3_req_id, how='left', on=['qa_user_id'])

                    in_combined_3 = []
                    for i in dataset1.index:
                        in_combined_3 += [True] if (pd.notnull(dataset1.loc[i, 'request_list_all3']) and dataset1['request_id'][i] in dataset1['request_list_all3'][i]) else [False]

                    dataset1['in_combined_3'] = in_combined_3

                    all_10_tags = dataset1[dataset1['in_combined_3']==False]

                    # final users with request count list
                    exact_first = exact_first_position[['qa_user_id', 'request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                    exact_first['price'] = exact_first['request_id']*600
                    exact_first.columns = ['qa_user_id','exact_first_position','exact_price']

                    invalid_req = invalid_requests[['qa_user_id', 'request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                    invalid_req['price'] = invalid_req['request_id']*600
                    invalid_req.columns = ['qa_user_id','invalid_requests','invalid_price']

                    neg2_req = neg2_requests[['qa_user_id', 'request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                    neg2_req['price'] = neg2_req['request_id']*1800
                    neg2_req.columns = ['qa_user_id','tagged_minus2','neg2_price']

                    tagged_all_10 = all_10_tags[['qa_user_id', 'request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                    tagged_all_10['price'] = tagged_all_10['request_id']*1200
                    tagged_all_10.columns = ['qa_user_id','tagged_all_10','all10_price']

                    final_df = pd.merge(exact_first, invalid_req, how='outer', on=['qa_user_id'])
                    final_df = pd.merge(final_df, neg2_req, how='outer', on=['qa_user_id'])
                    final_df = pd.merge(final_df, tagged_all_10, how='outer', on=['qa_user_id'])
                    final_df.fillna(0, inplace=True)
                    final_df['total_count'] = final_df['exact_first_position'] + final_df['invalid_requests'] + final_df['tagged_minus2'] + final_df['tagged_all_10']
                    final_df['total_price'] = final_df['exact_price'] + final_df['invalid_price'] + final_df['neg2_price'] + final_df['all10_price']

                    invalid_user_mistakes = invalid_requests[invalid_requests['index']!=-1][['qa_user_id','request_id']].groupby('qa_user_id', as_index=False).nunique('request_id')
                    invalid_user_mistakes['request_id'] = 'Found'
                    invalid_user_mistakes.columns = ['qa_user_id','invalid_requests']

                    exact_reqs = exact_first_position['request_id'].values
                    exact_first_mistakes = dataset1[(dataset1['request_id'].isin(exact_reqs)) & (~dataset1['index'].isin([0,-1]))][['qa_user_id','request_id']].groupby(['qa_user_id'], as_index=False).nunique('request_id')
                    exact_first_mistakes['request_id'] = 'Found'
                    exact_first_mistakes.columns = ['qa_user_id','exact_first_position']

                    negative2 = neg2_requests['request_id'].values
                    neg2_mistakes = dataset1[(dataset1['request_id'].isin(negative2)) & (dataset1['index'] >= 0) & (dataset1['feedback_tags'] >= 4)][['qa_user_id','request_id']].groupby('qa_user_id', as_index=False).nunique('request_id')
                    neg2_mistakes['request_id'] = 'Found'
                    neg2_mistakes.columns = ['qa_user_id','tagged_minus2']

                    final_mistakes = pd.merge(invalid_user_mistakes, exact_first_mistakes, how='outer', on='qa_user_id')
                    final_mistakes = pd.merge(final_mistakes, neg2_mistakes, how='outer', on='qa_user_id')
                    
                    logger.info("QA User Billing Finish")

                    # s3 file uploading
                    all_file_ids.append(curr_id)
                    current_file = pd.DataFrame(all_file_ids, columns=['file_id'])

                    output_file_names = [f'qa_user_request_count_{curr_id}.csv', 'processed_file_id.csv', f'users_not_following_rule_{curr_id}.csv']
                    output_files = [final_df, current_file, final_mistakes]

                    for i in range(3):
                        destination_file = destination_prefix + output_file_names[i]
                        utils.upload_csv_to_s3(destination_bucket, output_files[i], destination_file)
                    
                    # 
                    st.text(f'File ID : {curr_id}')
                    processed_user = len(final_df['qa_user_id'])
                    st.text(f'Process User : {processed_user}')
                    st.text(f'Error : {read_fail}')

                    logger.info(f'File ID : {curr_id}')
                    processed_user = len(final_df['qa_user_id'])
                    logger.info(f'Process User : {processed_user}')
                    logger.error(f'Error : {read_fail}')

                except Exception as e:
                    st.error(f'Error Processing QA Billing : {e}')
                    logger.error(f"Error Processing QA Billing : {e}")
                
                val_log = tail.contents() # extracting the log 

                # deleting all loggin variable for the current process
                log_handler.close()
                logging.shutdown()
                logger.removeHandler(log_handler)
                del logger, log_handler

                # saving the log file to S3
                try:
                    log_filename = f"msd_data_log_{curr_id}.txt" # the name of the log file
                    client.put_object(Bucket=destination_bucket, Key=destination_prefix + log_filename, Body=val_log)
                    print(destination_prefix + log_filename)
                    print(val_log)
                except Exception as e:
                    print(e)

    # with col2:
    #     st.download_button('Download RAW', utils.convert_df(raw_dataset), file_name=f'RAW_MSD_{qa_proccess_type}_{start_date} to {end_date}.csv', mime='text/csv', key='raw')


    # === DOWNLOAD SECTION ===
    st.header("")
    dwn_file_id = st.text_input("Type in file ID to download processed file", key="dlqa")
    dwn_file_type = st.radio("Choose file to download", ['QA User Request Count', 'Users Not Following Rule'], index=0)

    if dwn_file_id != "":
        try:
            if dwn_file_type == 'QA User Request Count':
                dwn_file_name = f'qa_user_request_count_{dwn_file_id}.csv'
            elif dwn_file_type == 'Users Not Following Rule':
                dwn_file_name = f'users_not_following_rule_{dwn_file_id}.csv'
            else:
                dwn_file_name = None
                raise Exception("Wrong Option")
                
            dwn_file = destination_prefix + dwn_file_name
            obj = client.get_object(Bucket= destination_bucket, Key= dwn_file)
            dwn_data = pd.read_csv(obj['Body']) # 'Body' is a key word
            csv = utils.convert_df(dwn_data)

            st.download_button(
                label="Download",
                data=csv,
                file_name=dwn_file_name,
                mime='text/csv',
            )

        except:
            st.error("File ID not found in S3")



with tab2: # === MSD Data File Processing ===
    # two columns for the start date and end date
    col1, col2 = st.columns(2)
    with col1:
        date_selector_proc = st.date_input(
            label="Enter Date",
            value=datetime.datetime.today() - datetime.timedelta(days=1),
            min_value=datetime.datetime.today() - datetime.timedelta(days=90),
            max_value=datetime.datetime.today() - datetime.timedelta(days=1),
            key='start_date_proc'
            )

    with col2:
        st.text("")
        st.text("")
        if st.button('Process', key='process_msd'):
            with st.spinner('Loading Files'):
                st.text("Reading Source MSD file")

                destination_file = destination_prefix + 'processed_file_id.csv'
                obj = client.get_object(Bucket=destination_bucket, Key=destination_file)
                previous_file = pd.read_csv(obj['Body'])  # 'Body' is a key word

                all_file_ids = list(previous_file['file_id'].values)
                last_file_id = all_file_ids[-1]
                curr_id = last_file_id+1

                # logging initialization
                logger = logging.getLogger("__process__") # creating logging variable
                logger.setLevel(logging.DEBUG) # set the minimun level of loggin to DEBUG
                formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s") # logging format that will appear in the log file
                tail = TailLogger(10000) # amount of log that is saved
                log_handler = tail.log_handler # variable log handler
                log_handler.setFormatter(formatter) # set formatter to handler
                logger.addHandler(log_handler) # adding file handler to logger
                
                try:
                    origin_key = origin_prefix+str(date_selector_proc)[0:10]+'/000.csv'
                    dataset = read_day(origin_key)

                    #"""## Data Clean-up"""

                    dataset['created_date'] = pd.to_datetime(dataset['created_date'])
                    dataset['feedback_tags1'] = dataset['extras.tags'].astype(str).apply(utils.remove_punctuations)
                    feedback_label = {'different': 0, '25 Similar': 1, '50 Similar': 2, '75 Similar': 3, '100 Similar': 4, 'exact': 5}
                    dataset['feedback_tags'] = dataset['feedback_tags1'].map(feedback_label)

                    # copy tagging data from index = -1 to all index
                    # search image info is only tagged in -1 index, rest of the indices contain recommended question tags
                    dataset['new_index'] = dataset['index']
                    # need to make the index for each row higher than -1 so that ffill can be applied on them after sorting by index value
                    dataset['new_index'][dataset['index'] == -2] = 10

                    # list of columns containing search image related tags
                    tag_columns = ['extras.flagged', 'extras.sample_type', 'extras.image_type',
                                'extras.picture_taken', 'extras.subject', 'extras.section',
                                'extras.chapter', 'extras.topic', 'extras.not_found_in_index']

                    # each request id can be QAed by as many as 3 QA users and each user may have different tags
                    # that's why grouping by request_id and qa_id is required followed by sorting by index for ffill
                    dataset.sort_values(by=['request_id', 'qa_user_id', 'new_index'], inplace=True)
                    dataset[tag_columns] = dataset.groupby(['request_id', 'qa_user_id'])[tag_columns].fillna(method='ffill')

                    dataset = dataset.drop('new_index', axis=1)

                    # keeping later date data for same (request_id and index) for same qa user
                    # qa user may tag something and then change the feedback, each click is registered in the sheet
                    dataset = dataset.groupby(['request_id', 'qa_user_id', 'index'], group_keys=False, as_index=False).apply(
                        lambda x: x.loc[x.created_date.idxmax()])

                    dataset['extras.subject'].fillna({i: "[]" for i in dataset.index}, inplace=True)
                    dataset['extras.subject'] = dataset['extras.subject'].apply(lambda x: list(set(ast.literal_eval(x))))
                    dataset['extras.subject'] = dataset['extras.subject'].astype(str).apply(utils.remove_punctuations)

                    # """
                    # conclusive data = where multiple qa users (max 3) QAed same (request_id & index pair) and >50% users similarity feedback matched
                    # inconclusive data = opposite of conclusive data
                    # """

                    # counting how many users QAed same request_id and index combination
                    user_count = dataset[['request_id', 'index', 'qa_user_id']].groupby(['request_id', 'index']).count().reset_index()
                    user_count.columns = ['request_id', 'index', 'qa_count']
                    # counting how many users gave same feedback for same request_id and index combination
                    feedback_count = dataset[['request_id', 'index', 'feedback_tags', 'qa_user_id']].groupby(
                        ['request_id', 'index', 'feedback_tags']).count().reset_index()
                    feedback_count.columns = ['request_id', 'index', 'feedback_tags', 'feedback_count']
                    # for crowdsourcing per request = 3
                    final_df = pd.merge(dataset, user_count, how='left', left_on=['request_id', 'index'], right_on=['request_id', 'index'])
                    final_df = pd.merge(final_df, feedback_count, how='left', left_on=['request_id', 'index', 'feedback_tags'],
                                        right_on=['request_id', 'index', 'feedback_tags'])
                    final_df["conc_pair"] = final_df["request_id"].astype(str) + "+index" + final_df["index"].astype(str)
                    conclusive = final_df[(final_df['index'] > -1) & (final_df['feedback_count'] / final_df['qa_count'] > 0.5)]
                    conclusive_pair = conclusive.drop_duplicates(subset='conc_pair')['conc_pair'].values
                    # inconclusive_data = final_df[~final_df['conc_pair'].isin(conclusive_pair)]
                    # inconclusive_data = inconclusive_data[inconclusive_data['index'] > -1]
                    final_df["is_conclusive"] = final_df["conc_pair"].map(lambda x: x in conclusive_pair)

                    final_df_conc = final_df[final_df["is_conclusive"]==True].drop_duplicates(subset='conc_pair')
                    final_df_inconc = final_df[final_df["is_conclusive"]==False].drop_duplicates(subset='conc_pair')
                    combined = pd.concat([final_df_conc, final_df_inconc], axis=0).drop_duplicates(subset='conc_pair')


                    # """## Text Extract
                    # Extracting text and doing similarity search only for negative set to count how many 
                    # exact matches from search in index has potential to be found by MSD model
                    # """

                    positive_set_requests = \
                        combined[(combined['feedback_tags'] >= 3) & (combined['index'] >= 0)].drop_duplicates(subset='request_id')[
                            'request_id'].values

                    text_df = combined[(~combined['request_id'].isin(positive_set_requests))]
                    text_df = text_df[['request_id', 'extras.subject', 'extras.searchImageUrl']].drop_duplicates().reset_index(drop=True)

                    files = text_df['extras.searchImageUrl'].values
                    batch_count = 10
                    st.text("OCR set length : " + str(len(files)))
                    logger.info("OCR set length : " + str(len(files)))


                    
                    start_time = datetime.datetime.now()
                    st.text(1)
                    img_files = asyncio.run(utils.get_img(files, batch_count))  # this is for running locally
                    st.text(2)
                    result = Parallel(n_jobs=1, prefer='threads', verbose=10)(delayed(utils.extract_text)(i, reader) for i in img_files[0:])
                    st.text(datetime.datetime.now() - start_time)
                    logger.info(datetime.datetime.now() - start_time)
                    st.text(3)

                    text_df['bounding_box'] = [i[0] for i in result]
                    text_df['extracted_text'] = [i[1] for i in result]
                    text_df['confidence'] = [i[2] for i in result]
                    text_df['min_confidence'] = [min(i[2]) if i[2] != ['Nothing'] else 'Nothing' for i in result]
                    text_df['max_confidence'] = [max(i[2]) if i[2] != ['Nothing'] else 'Nothing' for i in result]
                    text_df['avg_confidence'] = [np.mean(i[2]) if i[2] != ['Nothing'] else 'Nothing' for i in result]
                    text_df['extracted_full_text'] = text_df['extracted_text'].apply(lambda x: ' '.join(x))
                    st.text("OCR done")
                    logger.info("OCR done")

                    #"""## Text Embedding Similarity"""

                    corpus_bucket = st.secrets["destination_bucket"]
                    corpus_prefix = "current-ocr-file/"

                    file_name = 'physics_corpus_embeddings' + '.pkl'
                    file_key = corpus_prefix + file_name
                    physics_embeddings = utils.get_embedding_from_s3(corpus_bucket, file_key)

                    file_name = 'chemistry_corpus_embeddings' + '.pkl'
                    file_key = corpus_prefix + file_name
                    chemistry_embeddings = utils.get_embedding_from_s3(corpus_bucket, file_key)

                    file_name = 'maths_corpus_embeddings' + '.pkl'
                    file_key = corpus_prefix + file_name
                    maths_embeddings = utils.get_embedding_from_s3(corpus_bucket, file_key)

                    file_name = 'text_extracted_' + 'Maths' + '.csv'
                    file_key = corpus_prefix + file_name
                    maths = utils.get_csv_from_s3(corpus_bucket, file_key)

                    file_name = 'text_extracted_' + 'Physics' + '.csv'
                    file_key = corpus_prefix + file_name
                    physics = utils.get_csv_from_s3(corpus_bucket, file_key)

                    file_name = 'text_extracted_' + 'Chemistry' + '.csv'
                    file_key = corpus_prefix + file_name
                    chemistry = utils.get_csv_from_s3(corpus_bucket, file_key)

                    st.text(f'chemistry_file : {chemistry.shape}, maths_file : {maths.shape}, physics_file : {physics.shape}')
                    logger.info(f'chemistry_file : {chemistry.shape}, maths_file : {maths.shape}, physics_file : {physics.shape}')
                    st.text(
                        f'chemistry_embeddings : {chemistry_embeddings.shape}, maths_embeddings : {maths_embeddings.shape}, physics_embeddings : {physics_embeddings.shape}')
                    logger.info(f'chemistry_embeddings : {chemistry_embeddings.shape}, maths_embeddings : {maths_embeddings.shape}, physics_embeddings : {physics_embeddings.shape}')

                    physics_query = text_df[text_df['extras.subject'] == 'physics']
                    chemistry_query = text_df[text_df['extras.subject'] == 'chemistry']
                    maths_query = text_df[text_df['extras.subject'] == 'maths']

                    physics_queries = physics_query['extracted_full_text'].values
                    chemistry_queries = chemistry_query['extracted_full_text'].values
                    maths_queries = maths_query['extracted_full_text'].values

                    embedder1 = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                    embedder2 = SentenceTransformer('sentence-transformers/LaBSE')

                    maths_query_embeddings = utils.text_to_embeddings(embedder1, embedder2, maths_queries)
                    physics_query_embeddings = utils.text_to_embeddings(embedder1, embedder2, physics_queries)
                    chemistry_query_embeddings = utils.text_to_embeddings(embedder1, embedder2, chemistry_queries)

                    gc.collect()

                    st.text("physics semantic search")
                    logger.info("physics semantic search")
                    hits_physics = util.semantic_search(physics_query_embeddings, physics_embeddings, top_k=10)
                    physics_query['recommendations'] = hits_physics
                    physics_query['recommendation_question_id'] = physics_query['recommendations'].apply(
                        lambda x: [physics['questionId'].values[i['corpus_id']] for i in x])
                    physics_query['index_0_score'] = physics_query['recommendations'].apply(lambda x: [i['score'] for i in x][0])

                    st.text("chemistry semantic search")
                    logger.info("chemistry semantic search")
                    hits_chemistry = util.semantic_search(chemistry_query_embeddings, chemistry_embeddings, top_k=10)
                    chemistry_query['recommendations'] = hits_chemistry
                    chemistry_query['recommendation_question_id'] = chemistry_query['recommendations'].apply(
                        lambda x: [chemistry['questionId'].values[i['corpus_id']] for i in x])
                    chemistry_query['index_0_score'] = chemistry_query['recommendations'].apply(lambda x: [i['score'] for i in x][0])

                    st.text("maths semantic search")
                    logger.info("maths semantic search")
                    hits_maths = util.semantic_search(maths_query_embeddings, maths_embeddings, top_k=10)
                    maths_query['recommendations'] = hits_maths
                    maths_query['recommendation_question_id'] = maths_query['recommendations'].apply(
                        lambda x: [maths['questionId'].values[i['corpus_id']] for i in x])
                    maths_query['index_0_score'] = maths_query['recommendations'].apply(lambda x: [i['score'] for i in x][0])

                    text_df1 = pd.concat([maths_query, physics_query, chemistry_query], axis=0)

                    text_df1['word_count'] = text_df1['extracted_full_text'].apply(lambda words: len(words.split()))
                    text_df1['avg_word_len'] = text_df1['extracted_full_text'].apply(
                        lambda words: sum(len(word) for word in words.split(' ')) / len(words.split(' ')))
                    text_df1['decile_rank'] = pd.qcut(text_df1['index_0_score'], 10, labels=False)
                    text_df1.drop('extras.subject', axis=1, inplace=True)

                    #"""## Final Files"""

                    final_df = pd.merge(combined, text_df1, how='left', on=['request_id', 'extras.searchImageUrl'])

                    # negative positive and QA sets
                    benchmarking_d0 = final_df[(final_df['index'] == 0) & (final_df['is_d0'] == 't') & (
                        final_df['extras.subject'].isin(['maths']))][
                                    0:10].reset_index(drop=True)
                    benchmarking_non_d0 = final_df[
                                            (final_df['extras.image_type'] == '[]') & (final_df['index'] == 0) & (
                                                        final_df['is_d0'] == 'f') & (
                                                final_df['extras.subject'].isin(['maths']))][0:50].reset_index(drop=True)
                    benchmarking = pd.concat([benchmarking_d0, benchmarking_non_d0], axis=0).reset_index(drop=True)
                    benchmarking = benchmarking[
                        ['request_created_date', 'request_id', 'extras.picture_taken', 'extras.subject', 'extras.section', 'extras.chapter',
                        'extras.topic', 'extras.searchImageUrl', 'is_d0', 'feedback_tags1']]
                    benchmarking.columns = ['Date', 'request_id', 'extras.picture_taken', 'extras.subject', 'extras.section',
                                            'extras.chapter', 'extras.topic', 'extras.searchImageUrl', 'is_d0', 'Colearn']

                    final_df = final_df.drop(['feedback_tags1'], axis=1)

                    positive_set_all = final_df[(final_df['feedback_tags'] >= 3) & (final_df['index'] >= 0)]

                    positive_requests = positive_set_all.drop_duplicates(subset='request_id')['request_id'].values

                    neg2_set = final_df[
                        (final_df['index'] == -2) & (final_df['feedback_tags'] >= 4) & (~final_df['request_id'].isin(positive_requests))]
                    neg2_set_req = neg2_set.drop_duplicates(subset='request_id')['request_id'].values
                    negative_set = final_df[
                        (final_df['feedback_tags'] < 3) & (final_df['index'] >= 0) & (~final_df['request_id'].isin(positive_requests))]
                    # 0 index of negative set will also be 0 and all negative set will have 0 index and that will be the best match
                    # remove -2 so as to not repeat search in index for those requests while sourcing
                    negative_set_0 = negative_set[(negative_set['index'] == 0) & (~negative_set['request_id'].isin(neg2_set_req))]

                    match_count = neg2_set[neg2_set['recommendation_question_id'].notnull()][
                        ['request_id', 'question_id', 'recommendation_question_id']]
                    match_count['question_id'] = match_count['question_id'].map(lambda x: x.split(' '))
                    match_count['Match'] = match_count.apply(
                        lambda x: len(set(x.question_id).intersection(set(x.recommendation_question_id))) > 0, axis=1)

                    neg2_set = pd.merge(neg2_set, match_count, how='left', on='request_id')

                    st.text(match_count['Match'].value_counts())
                    logger.info(match_count['Match'].value_counts())

                    st.text(f'File ID : {curr_id}')

                    all_file_ids.append(curr_id)
                    current_file = pd.DataFrame(all_file_ids, columns=['file_id'])

                    #"""## Write processed file to S3"""
                    all_files = [final_df, benchmarking, positive_set_all, negative_set, neg2_set, negative_set_0, current_file]
                    file_names = [
                        f'MSD_QC_Processed_{curr_id}.csv',
                        f'QC_benchmark_{curr_id}.csv', 
                        f'positive_{curr_id}.csv', 
                        f'negative_{curr_id}.csv',
                        f'negative2_{curr_id}.csv', 
                        f'negative_set_0_{curr_id}.csv',
                        'processed_file_id.csv'
                        ]

                    destination_bucket = st.secrets["destination_bucket"]
                    destination_prefix = st.secrets["destination_prefix"]

                    for i in range(len(all_files)):
                        destination_file = destination_prefix + file_names[i]
                        utils.upload_csv_to_s3(destination_bucket, all_files[i], destination_file)
                
                except Exception as e:
                    st.error(f'Error Processing MSD Data : {e}')
                    logger.error(f"Error Processing MSD Data : {e}")

                val_log = tail.contents() # extracting the log 

                # deleting all loggin variable for the current process
                log_handler.close()
                logging.shutdown()
                logger.removeHandler(log_handler)
                del logger, log_handler

                # saving the log file to S3
                try:
                    log_filename = f"msd_data_log_{curr_id}.txt" # the name of the log file
                    client.put_object(Bucket=destination_bucket, Key=destination_prefix + log_filename, Body=val_log)
                    print(destination_prefix + log_filename)
                    print(val_log)
                except Exception as e:
                    print(e)



    # === DOWNLOAD SECTION ===
    st.header("")
    dwn_file_id = st.text_input("Type in file ID to download processed file", key='dlpc')
    dwn_file_type = st.radio("Choose file to download", ['MSD_QC_Processed', 'QC_benchmark', 'positive', 'negative',
              'negative2', 'negative_set_0'], index=0)

    if dwn_file_id != "":
        try:    
            dwn_file = destination_prefix + dwn_file_type + "_" + dwn_file_id + ".csv"
            obj = client.get_object(Bucket= destination_bucket, Key= dwn_file)
            dwn_data = pd.read_csv(obj['Body']) # 'Body' is a key word
            csv = utils.convert_df(dwn_data)

            st.download_button(
                label="Download",
                data=csv,
                file_name=dwn_file_name,
                mime='text/csv',
            )

        except:
            st.error("File ID not found in S3")