from typing import Dict, Literal
from decimal import Decimal
import datetime

from suql import suql_execute
from suql.postgresql_connection import execute_sql_with_column_info, apply_auto_limit
import os
from dotenv import load_dotenv

import mysql.connector
import requests
import aiomysql
import ssl

from aiocache import caches, cached
from aiomysql import create_pool
import asyncio

# Configure the default cache to use Redis
caches.set_config({
    'default': {
        'cache': "aiocache.RedisCache",
        'endpoint': "localhost",
        'port': 6379,
        'serializer': {
            'class': "aiocache.serializers.JsonSerializer"
        },
        'plugins': []
    }
})


# Dictionary to hold MySQL connections
mysql_connections = {}

mysql_connections_lock = asyncio.Lock()

async def get_mysql_connection(db_secrets_file):
    global mysql_connections

    # Load environment variables from the specified .env file
    load_dotenv(db_secrets_file)

    # Retrieve database credentials from environment variables
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    ssl_ca = os.getenv('SSL_CA')
    database = os.getenv('DATABASE')

    connection_key = (host, port, user, database)

    async with mysql_connections_lock:
        pool = mysql_connections.get(connection_key)
        if pool is None:
            print(f"using {db_secrets_file} to establish connection")
            ssl_context = None
            if ssl_ca:
                ssl_context = ssl.create_default_context(cafile=ssl_ca)
        
            pool = await create_pool(
                host=host,
                port=int(port),
                user=user,
                password=password,
                db=database,
                ssl=ssl_context,
                minsize=1,
                maxsize=20,  # Adjust pool size as needed
            )
            mysql_connections[connection_key] = pool

    return pool

def prepare_initialize(table_schema):
    import pandas as pd
    # table_schema is passed in as a CSV file
    if type(table_schema) is str and os.path.exists(table_schema):
        with open(table_schema, 'r') as file:
            if file.read().strip():
                df = pd.read_csv(table_schema)
                table_schema_lst = df
                table_w_ids = {}
                for _, row in df.iterrows():
                    table_name = row['table_name']
                    table_w_ids[table_name] = row['id_field_name']
                return table_w_ids, table_schema_lst
            else:
                return {}, []
    else:
        raise ValueError()

def convert_sql_result_to_dict(results, column_names):
    if not results:
        return results
    
    data = []
    for row in results:
        row_data = {}
        for col_index, col_value in enumerate(row):
            if isinstance(col_value, Decimal):
                col_value = float(col_value)
            if isinstance(col_value, datetime.date):
                col_value = col_value.strftime('%Y-%m-%d')
            row_data[column_names[col_index]] = col_value
        data.append(row_data)

    return data

@cached(ttl=3600)
async def execute_sql(
    sql: str,
    table_w_ids: Dict = {},
    database_name: str = "ingestion_experimental",
    suql_model_name="gpt-4-turbo",
    embedding_server_address: str = "http://127.0.0.1:8509",
    source_file_mapping: Dict = {},
    db_type: Literal["postgres", "mysql", "sparql"] = "postgres",
    db_secrets_file = None,
    suql_enabled = False,
    limit_query = " LIMIT 1000",
):
    if db_type == "postgres":
        try:
            if suql_enabled:
                sql = apply_auto_limit(sql, limit_query= " LIMIT 10")
                results, column_names, _ = suql_execute(
                    sql,
                    table_w_ids,
                    database_name,
                    embedding_server_address=embedding_server_address,
                    source_file_mapping=source_file_mapping,
                    llm_model_name=suql_model_name,
                    disable_try_catch=True,
                    disable_try_catch_all_sql=True,
                )
            else:
                results, column_names = execute_sql_with_column_info(
                    apply_auto_limit(sql),
                    database=database_name,
                    unprotected=True
                )
                column_names = list(map(lambda x:x[0], column_names))
            status = None
            results = convert_sql_result_to_dict(
                results, column_names
            )
        except Exception as e:
            results = None
            column_names = None
            status = str(e)
            print(f"error executing PostgreSQL query: {status}")

    elif db_type == "mysql":
        try:
            if limit_query:
                sql = apply_auto_limit(sql, limit_query=limit_query)
                
            pool = await get_mysql_connection(db_secrets_file)
            async with pool.acquire() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(sql)
                    results = await cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    status = None
            
            results = convert_sql_result_to_dict(
                results, column_names
            )
        except (aiomysql.Error, TypeError) as err:
            results = None
            column_names = None
            status = str(err)
        
    elif db_type == "sparql" and database_name in ["datacommons"]:
        load_dotenv(db_secrets_file)
        url = "https://api.datacommons.org/v2/sparql"
        headers = {
            "X-API-Key": os.getenv('DATACOMMONS_API_KEY'),
            "Content-Type": "application/json"
        }
        data = {
            "query": sql
        }
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()

        results = []
        if "rows" not in response_json:
            if "header" in response_json:
                return "no results found", "no results found"
            return str(response_json), str(response_json)
        
        for d in response_json["rows"]:
            new_d = {}
            for h, value_d in zip(response_json["header"], d["cells"]):
                new_d[h] = value_d["value"]
            results.append(new_d)
        column_names = response_json["header"]

        status = None
    else:
        raise ValueError()

    return results, status


def search_graph(
    search_string: str,
    db_type: Literal["postgres", "mysql", "sparql"] = "sparql",
    database_name: str = "datacommons",
    db_secrets_file = None,
):
    if db_type == "sparql" and database_name in ["datacommons"]:
        load_dotenv(db_secrets_file)
        url = "https://api.datacommons.org/v2/resolve"
        headers = {
            "X-API-Key": os.getenv('DATACOMMONS_API_KEY'),
            "Content-Type": "application/json"
        }
        data = {
            "nodes": [search_string] if type(search_string) == str else search_string,
            "property": "<-description->dcid"
        }
        response = requests.post(url, headers=headers, json=data)
        res = response.json()
        formatted_res = []
        if "entities" in res:
            for i in res["entities"]:
                if "resolvedIds" in i:
                    formatted_res.append({
                        "search_str": i['node'],
                        "resolvedIds": i["resolvedIds"]
                    })
                else:
                    formatted_res.append({
                        "search_str": i['node'],
                        "resolvedIds": []
                    })
            return formatted_res
            
        return "no results found"
    else:
        raise ValueError()
    
def graph_fetch_incoming_outgoing_edges(
    node_id: str,
    db_type: Literal["postgres", "mysql", "sparql"] = "sparql",
    database_name: str = "datacommons",
    db_secrets_file = None,
):
    """
    Specific for graph databases. Retrieve the outgoing and incoming edge (property) names
    of a given database.
    """
    if db_type == "sparql" and database_name in ["datacommons"]:
        load_dotenv(db_secrets_file)
        url = "https://api.datacommons.org/v2/node"
        headers = {
            "X-API-Key": os.getenv('DATACOMMONS_API_KEY'),
            "Content-Type": "application/json"
        }
        outgoing_data = {
            "nodes": [node_id] if type(node_id) == str else node_id,
            "property": "->"
        }
        res = {}
        outgoing_response = requests.post(url, headers=headers, json=outgoing_data)
        outgoing_result = outgoing_response.json()
        
        res["outgoing_properties"] = []
        if "data" in outgoing_result and node_id in outgoing_result["data"]:
            res["outgoing_properties"] = outgoing_result["data"][node_id]
        
        incoming_data = {
            "nodes": [node_id],
            "property": "<-"
        }
        incoming_response = requests.post(url, headers=headers, json=incoming_data)
        incoming_result = incoming_response.json()
        
        res["incoming_properties"] = []
        if "data" in incoming_result and node_id in incoming_result["data"]:
            res["incoming_properties"] = incoming_result["data"][node_id]
        
        return res
        
    else:
        raise ValueError()

# Establishes connection to MySQL DB
def connect_to_mysql(host, port, user, password, database, ssl_ca):
    connection = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,  
        ssl_ca=ssl_ca 
    )
    if connection.is_connected():
        print("Connected to MySQL database!")
    else:
        print("Failed to connect.")
    
    return connection