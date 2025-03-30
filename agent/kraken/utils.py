import asyncio
import os
import warnings
import json
from typing import List
import math
from pydantic import BaseModel
from collections import defaultdict

from chainlite import chain, get_logger, llm_generation_chain, load_config_from_file
from json_repair import repair_json

from kraken.state import SqlQuery
from kraken.sql_utils import connect_to_mysql, execute_sql
from dotenv import load_dotenv

import random
import re
import requests
import ast

import asyncio
import redis.asyncio as redis
import hashlib
import pickle

# Set to a really high number if not worried about too many LLM calls
MAX_SHARDS_TO_PROCESS = 300 # will only process this many shards from the column (might not process whole column if this is too low)

logger = get_logger(__name__)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
load_config_from_file(os.path.join(CURRENT_DIR, "..", "llm_config.yaml"))

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)  # from ReFinED


class BaseParser:
    @classmethod
    def initialize(engine: str):
        raise NotImplementedError("Subclasses should implement this method")

    @classmethod
    def run_batch(cls, questions: list[str]):
        return asyncio.run(
            cls.runnable.with_config(
                {"recursion_limit": 60, "max_concurrency": 50}
            ).abatch(questions)
        )


@chain
async def parse_string_to_json(output: str) -> dict:
    return repair_json(output, return_objects=True)


@chain
def extract_code_block_from_output(llm_output: str, code_block: str) -> str:
    code_block = code_block.lower()
    if f"```{code_block}" in llm_output.lower():
        start_idx = llm_output.lower().rfind(f"```{code_block}") + len(
            f"```{code_block}"
        )
        end_idx = llm_output.lower().rfind("```", start_idx)
        if end_idx < 0:
            # because llm_generation_chain does not include the stop token
            end_idx = len(llm_output)
        extracted_block = llm_output[start_idx:end_idx].strip()
        return extracted_block
    else:
        raise ValueError(f"Expected a code block, but llm output is {llm_output}")


@chain
def sql_string_to_sql_object(sql: str, table_w_ids: dict, database: str, embedding_server_address: str, source_file_mapping: str, suql_model_name: str, db_type: str, db_secrets_file: str) -> SqlQuery:
    return SqlQuery(
        sql=sql,
        table_w_ids=table_w_ids,
        database_name=database,
        embedding_server_address=embedding_server_address,
        source_file_mapping=source_file_mapping,
        suql_model_name=suql_model_name,
        db_type=db_type,
        db_secrets_file=db_secrets_file
    )


@chain
def execute_sql_object(
    sql: SqlQuery,
) -> str:
    sql.execute()
    return sql


def format_table_schema(schema: str) -> str:
    # TODO: Used to format the table schemas from sql, right now we just return the schema as is.
    return schema


def get_examples(utterance: str, examples: list[str]) -> list[str]:
    top_examples = rerank_list.bind(top_k=3).ainvoke(
        {
            "query_text": utterance,
            "item_list": examples,
        }
    )
    return top_examples


def process_reranking_output(response):
    new_response = ""
    for character in response:
        if not character.isdigit():
            new_response += " "
        else:
            new_response += character
    response = new_response.strip()
    response = [int(x) - 1 for x in response.split()]

    # deduplicate
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)

    return new_response


async def llm_rerank_window(query_text, retrieval_results):
    reranking_prompt_output = await llm_generation_chain(
        template_file="rerank_list.prompt",
        engine="gpt-4o",
        max_tokens=700,
        keep_indentation=True,
    ).ainvoke(
        {
            "query": query_text,
            "retrieval_results": retrieval_results,
        }
    )

    reranked_indices = process_reranking_output(reranking_prompt_output)
    reranked_indices = [
        i for i in reranked_indices if i < len(retrieval_results)
    ]  # remove numbers that are outside the range
    logger.debug("reranked_indices = %s", str(reranked_indices))
    return reranked_indices, reranking_prompt_output


@chain
async def rerank_list(query_text, item_list, top_k=3):
    llm_reranker_sliding_window_size = 20
    llm_reranker_sliding_window_step = 10

    end_index = len(item_list)
    start_index = end_index - llm_reranker_sliding_window_size
    original_rank = list(range(len(item_list)))
    while True:
        if start_index < 0:
            start_index = 0
        reranked_indices, reranking_prompt_output = await llm_rerank_window(
            query_text, item_list[start_index:end_index]
        )

        if len(reranked_indices) != (end_index - start_index):
            missing_indices = set(range(end_index - start_index))
            for found_index in reranked_indices:
                if found_index in missing_indices:
                    missing_indices.remove(found_index)

            logger.warning(
                "LLM reranking should return the same number of outputs as inputs. Adding missing indices: %s. Prompt output was %s",
                str(missing_indices),
                reranking_prompt_output,
            )

            # TODO instead of adding missing indices, shift everything and continue
            # Add missing indices to the end so that we don't crash
            # This is reasonable assuming that if the LLM did not output an index, it probably was not that relevant to the query to begin with
            reranked_indices = reranked_indices + list(missing_indices)

            assert len(reranked_indices) == (end_index - start_index)

        item_list[start_index:end_index] = [
            item_list[start_index + reranked_indices[i]]
            for i in range(len(reranked_indices))
        ]
        original_rank[start_index:end_index] = [
            original_rank[start_index + reranked_indices[i]]
            for i in range(len(reranked_indices))
        ]

        if start_index == 0:
            break
        end_index -= llm_reranker_sliding_window_step
        start_index -= llm_reranker_sliding_window_step

    return item_list[:top_k]

def extract_psql_comments(sql):
    # Split the SQL into lines
    lines = sql.strip().split('\n')

    comments = []
    sql_without_comments = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('--'):
            # Collect comments
            comment = stripped_line.lstrip('- ').rstrip()
            comments.append(comment)
        else:
            sql_without_comments.append(line)

    # Reconstruct the SQL without comments
    sql_no_comments = '\n'.join(sql_without_comments)

    return '\n'.join(comments)


def get_tables(schema) -> dict:
    non_enum_schemas = schema[schema.apply(lambda x: x["type"] != "enum", axis=1)]
    res = non_enum_schemas[["table_name", "table_CREATE_command"]].to_dict(orient='records')
    for i in res:
        i["comments"] = extract_psql_comments(i["table_CREATE_command"])
        del i["table_CREATE_command"]
    
    return res

def retrieve_tables_details(utterance: List[str], schema) -> dict:
    res = schema[schema.apply(lambda x:x["table_name"] in utterance, axis=1)]
    
    return res["table_CREATE_command"].astype(str).tolist()

async def llm_select_entities_in_batches_async(
    search_str, 
    all_distinct_values, 
    shard_size=128, 
    num_shuffled_iters=1, 
    voting_threshold=1
    ):
    all_distinct_values_set = set(all_distinct_values)

    shards = [
        all_distinct_values[i:i + shard_size]
        for i in range(0, len(all_distinct_values), shard_size)
    ][:MAX_SHARDS_TO_PROCESS]

    entity_candidates_dict = defaultdict(int)

    for round in range(num_shuffled_iters):

        shard_args = [
            {
                'search_string':search_str,
                'distinct_entities':shard
            } 
            for shard in shards
        ]
        # print('Shard args sample is are:', shard_args[:5])
        responses = await llm_generation_chain(
            template_file="select_entities.prompt",
            engine="gpt-4o-mini",
            max_tokens=3000,
            temperature=0.0
        ).abatch(shard_args) 

        total_filtered = []
        for filtered_shard in responses:
            #total_filtered.extend(ast.literal_eval(filtered_shard))
            try:
                total_filtered.extend(repair_json(filtered_shard, return_objects=True))
            except:
                ValueError("Failed to convert LLM NED response into list")   

        for entity in total_filtered:
            entity_candidates_dict[entity] += 1
        
        if round + 1 < num_shuffled_iters:
            for i in range(len(shards)):
                random.shuffle(shards[i])
        
    # filtering out entities that the LLM didn't pick enough AND were hallucinated and don't actually exist in DB
    final_candidates = [
        entity 
        for entity, count in entity_candidates_dict.items()
        if count >= voting_threshold and 
        entity in all_distinct_values_set
    ]

    # print(f'Final candidates sample are: {final_candidates[:50]}')

    final_candidates = set(final_candidates)
    filtered_out_candidates = [candidate for candidate in total_filtered if candidate not in final_candidates]
    
    # print(f'The candidates that were insufficiently voted on OR hallucinated is {filtered_out_candidates}')
    
    return final_candidates

class RedisLLMCacheEntityLinking:
    _instance = None
    _locks = {}  # Class-level locks shared across all instances

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RedisLLMCacheEntityLinking, cls).__new__(cls)
        return cls._instance

    def __init__(self, redis_url="redis://localhost:6379", ttl=3600):
        if not hasattr(self, 'redis'):
            self.redis = redis.from_url(redis_url, decode_responses=False)  # Binary mode for pickled objects
            self.ttl = ttl

    @classmethod
    async def _hash_key(cls, input_text):
        """Generate a unique key for caching based on input text"""
        return hashlib.sha256(input_text.encode()).hexdigest()

    async def entity_linking_caching(
        self, 
        search_str,
        col_name_lst,
        db_secrets_file,
        location_cols = ['stanford_api_data.location', 'stanford_api_data.country', 'stanford_api_data.region', 'stanford_api_data.iso'],
        free_text_cols = ['stanford_api_data.notes'],
        db_type = 'mysql'
    ):
        """Cached LLM call with Redis"""
        input_text = f"{search_str}-{','.join(col_name_lst)}-{db_secrets_file}-{','.join(location_cols)}-{','.join(free_text_cols)}-{db_type}"
        key = await self._hash_key(input_text)
        cached_result = await self.redis.get(key)

        if cached_result:
            return pickle.loads(cached_result)  # Deserialize and return cached response

        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:  # Ensures only one concurrent request per unique input
            cached_result = await self.redis.get(key)  # Double-check cache
            if cached_result:
                return pickle.loads(cached_result)

            # Simulated long-running LLM call
            result = await entity_linking(
                search_str=search_str,
                col_name_lst=col_name_lst,
                db_secrets_file=db_secrets_file,
                location_cols=location_cols,
                free_text_cols=free_text_cols,
                db_type=db_type
            )

            # Store the result in cache
            await self.redis.setex(key, self.ttl, pickle.dumps(result))
            del self._locks[key]  # Cleanup lock

        return result

async def entity_linking(
    search_str,
    col_name_lst,
    db_secrets_file,
    location_cols = ['stanford_api_data.location', 'stanford_api_data.country', 'stanford_api_data.region', 'stanford_api_data.iso'],
    free_text_cols = ['stanford_api_data.notes'],
    db_type = 'mysql'
):
    if set(location_cols) & set(col_name_lst):
        return f"'I should not use entity_linking on a location-related column. Instead, I should use location_linking action to resolve locations.'"
    elif set(free_text_cols) & set(col_name_lst):
        return f"'I should not use entity_linking on free-text columns. I should instead try to filter on different columns, only use free-text columns in projection, or use a LIKE operator on free-text column if neccesarry."  # possibly add option to allow %LIKE% operators in this ase
    
    output_entities = {}

    # Filter entities for each input column
    for col in col_name_lst:

        table_name, col_name = col.split('.')

        # get unique values from the column

        # TODO: consider caching this for each column since it takes some time
        distinct_query = f"SELECT DISTINCT {col_name} FROM {table_name};"
        # print('final distinct query for NED is:', distinct_query)
        result_dicts, _ = await execute_sql(distinct_query, db_type=db_type, limit_query=None, db_secrets_file=db_secrets_file)

        distinct_values = [dic[col_name] for dic in result_dicts]

        # do filtration
        ents = await llm_select_entities_in_batches_async(search_str, distinct_values)

        output_entities[col_name] = list(ents)
    return output_entities

async def postprocess_entities(query, substitution_dict):
    comment = ""
    if not substitution_dict:
        return query

    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, substitution_dict.keys())) + r')\b')
    revised_query = pattern.sub(lambda m: substitution_dict[m.group()], query)
    
    if revised_query != query:
        comment = "-- The agent's original query contained variable names to represent entities, which were expanded into the full list in post-processing."

    return comment + '\n' + revised_query

async def location_linking(location_lst):
    input_error_msg = "I encountered an error in resolving locations. I should try again, ensuring the input arguments adhere to requirements."

    async def get_location_conversion_info(location_options_dict):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # LLM resolves location
                get_location_chain = llm_generation_chain(
                    template_file="select_locations.prompt",
                    engine="gpt-4o",
                    max_tokens=5000,
                    temperature=0.0,
                )
                output = await get_location_chain.ainvoke(
                    {
                        "location_list": location_lst,
                        "database_type": 'mysql',
                        "location_options_dict": location_options_dict
                    }
                )
                
                # Parse json response
                try:
                    output = repair_json(output, return_objects=True)
                except Exception as json_error:
                    raise ValueError(f"Failed to convert LLM response into dict: {str(json_error)}")
                
                # Validate it is a dict
                if not isinstance(output, dict):
                    raise ValueError("LLM output is not a valid dictionary")
                
                # Validate all locations are present (if it missed one, retry)
                if set(output.keys()) != set(location_lst):
                    raise ValueError("The location_linker did not link locations for every location search string")
                
                break
            
            except Exception:
                if attempt == max_retries - 1:
                    output = input_error_msg
        
        return output


    def get_location_from_azure(query):
        EARTH_RADIUS = 6371000  # meters
        TOLERANCE = 1500  # meters

        subscription_key = os.environ["AZURE_MAP_KEY"]
        # API endpoint
        url = "https://atlas.microsoft.com/search/address/json"

        # Parameters for the request
        params = {
            "subscription-key": subscription_key,
            "api-version": "1.0",
            "language": "en-US",
            "query": query,
        }
        # Sending the GET request
        response = requests.get(url, params=params)

        if response.status_code != 200:
            return "Azure API request failed with status code {response.status_code}: {response.text}", 0

        # Extracting the JSON response
        response_json = response.json()

        if "results" not in response_json or not response_json["results"] or "type" not in response_json["results"][0] or 'boundingBox' not in response_json["results"][0]:
            return "The Azure API returned no results for the given location - I should try again with a different search string.",0

        if response_json["results"][0]["type"] == "Geography":
            bbox = response_json["results"][0]["boundingBox"]
            latitude_north, longitude_west = (
                bbox["topLeftPoint"]["lat"],
                bbox["topLeftPoint"]["lon"],
            )
            latitude_south, longitude_east = (
                bbox["btmRightPoint"]["lat"],
                bbox["btmRightPoint"]["lon"],
            )
        else:
            # get coords
            coord = response_json["results"][0]["position"]
            longitude, latitude = coord["lon"], coord["lat"]

            # Get location range
            delta_longitude = TOLERANCE / EARTH_RADIUS * 180 / math.pi
            delta_latitude = (
                TOLERANCE
                / (EARTH_RADIUS * math.cos(latitude / 180 * math.pi))
                * 180
                / math.pi
            )

            longitude_west = longitude - delta_longitude
            longitude_east = longitude + delta_longitude
            latitude_south = latitude - delta_latitude
            latitude_north = latitude + delta_latitude

        return ((longitude_west, longitude_east, latitude_south, latitude_north), 1)
    
    with open('location_options_dict.json', 'r') as file:
        location_options_dict = json.load(file)
    output = await get_location_conversion_info(location_options_dict)

    if output == input_error_msg:
        return input_error_msg

    # Else format responses with right coordinate substitution
    for loc in output:
        if not output[loc]:  # LLM deemed search string needing of lat/long coords
            coord_output = get_location_from_azure(loc)
            
            if not coord_output[1]:
                output[loc] = coord_output[0]
            else:
                longitude_west, longitude_east, latitude_south, latitude_north = coord_output[0]
                output[loc] = f"longitude BETWEEN {longitude_west} AND {longitude_east} AND latitude BETWEEN {latitude_south} AND {latitude_north}"
    
    return output