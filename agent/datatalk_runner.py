import re
import os
import json
import urllib.parse
from typing import Optional
from dotenv import load_dotenv
from kraken.state import Action
from kraken.sql_utils import prepare_initialize
from dataclasses import dataclass

from kraken.agent import DatatalkParser, retrieve_relevant_domain_specific_instructions
from kraken.utils import postprocess_entities
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl
from chainlite import write_prompt_logs_to_file
from decimal import Decimal

from langchain_core.prompts.string import jinja2_formatter

from openai import OpenAI
from pydantic import BaseModel
import os
import hashlib

import os
from pydantic import BaseModel

# Directly use OpenAI for streaming for only the reporter prompt
client = OpenAI()

load_dotenv()

from multiprocessing import Value
from threading import Lock
import litellm

global_cost_counter = Value('d', 0.0)  # 'd' for double precision float
counter_lock = Lock()

async def track_cost_callback(
    kwargs,                 # kwargs to completion
    completion_response,    # response from completion
    start_time, end_time    # start/end time
):
    try:
        response_cost = kwargs["response_cost"] # litellm calculates response cost for you
        # print("regular response_cost", response_cost)
        
        # Safely update the global counter
        with counter_lock:
            global_cost_counter.value += response_cost
            # print(f"Updated global cost counter: {global_cost_counter.value}")
    except Exception as e:
        print(f"track_func_token_callback encountered exception: {e}")
        pass
    
# This registers cost tracker on each litellm call, across different processes & threads
litellm.success_callback = [track_cost_callback]

async def on_chat_start(
    look_up_table_path,
    domain_specific_instructions,
    database
):  
    table_w_ids, table_schema_lst = prepare_initialize(look_up_table_path)

    suql_model_name = "azure/gpt-4o"
    embedding_server_address = "http://127.0.0.1:57378"
    db_type = "postgres"
    db_details_path = None
    db_secrets_file = None
    suql_enabled = False
    database_name = database
    if os.path.exists(db_details_path):
        with open(db_details_path, "r") as fd:
            db_details = json.load(fd)
        if "db_type" in db_details:
            db_type = db_details["db_type"]
        if "db_secrets_file" in db_details:
            db_secrets_file = db_details["db_secrets_file"]
        if "suql_enabled" in db_details:
            suql_enabled = db_details["suql_enabled"]
        if "database_name" in db_details:
            database_name = db_details["database_name"]

    available_actions = [
        'get_tables',
        'retrieve_tables_details',
        'execute_sql',
    ]

    DatatalkParser.initialize(
        engine="gpt-4o",
    )
    semantic_parser_class = DatatalkParser(
        engine="gpt-4o",
        table_w_ids=table_w_ids,
        database_name=database_name,
        suql_model_name=suql_model_name,
        embedding_server_address=embedding_server_address,
        table_schema=table_schema_lst,
        domain_specific_instructions=domain_specific_instructions,
        db_type=db_type,
        db_secrets_file=db_secrets_file,
        suql_enabled=suql_enabled,
        available_actions=available_actions
    )
    
    return semantic_parser_class

@dataclass
class SqlReporterResponse(BaseModel):
    generated_sql: Optional[str] # the last generated sql, only applicable if you succeed
    
    technical_explanation_of_SQL: Optional[str] # only applicable if you succeed
    
    simplified_explanation_of_SQL: Optional[str] # only applicable if you succeed
    
    limitations: Optional[str] # only applicable if you succeed
    
    report: Optional[str] # report, write to this field if user is not engaged in database-related conversation but instead just chit-chatting, or if you were unable to find the desired output.
    
    suggested_next_turn_natural_language_queries: Optional[list[str]] # suggest some next turn natural language queries as future exploration direction.
    

def is_list(obj):
    return isinstance(obj, list)

def is_dict(obj):
    return isinstance(obj, dict)

def _ensure_strict_json_schema(
    json_schema: object,
    path: tuple[str, ...],
):
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the API expects.

    Adapted from OpenAI's Python client code
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key)
            )
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"))

    # unions
    any_of = json_schema.get("anyOf")
    if is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)))
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if is_list(all_of):
        json_schema["allOf"] = [
            _ensure_strict_json_schema(entry, path=(*path, "anyOf", str(i)))
            for i, entry in enumerate(all_of)
        ]

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name))

    return json_schema

def map_step_name_to_natural_names(step_name: str):
    if step_name == "execute_sql":
        return "Building queries"
    elif step_name == "get_tables":
        return "Understand database structure"
    elif step_name == "retrieve_tables_details":
        return "Get table details"
    elif step_name == "entity_linking":
        return "Linking entities"
    elif step_name == "location_linking":
        return "Resolving locations"
    
    return step_name

def format_decimal(val):
    # Check if the value is a float or Decimal
    if isinstance(val, (float, Decimal)):
        return f"{val:,.2f}"  # Format to 2 decimal places and include commas
    return val  # Return as is if not a float or Decimal

def update_column_with_links(df):
    # Update the column values to the desired format
    pattern_committee = re.compile(r'^C\d{8}$')
    pattern_candidate = re.compile(r'^[SPH][\dA-Z]{8}$')

    # Iterate through each column in the dataframe
    for col in df.columns:
        # Check if all values in the column match the pattern
        if df[col].apply(lambda x: bool(pattern_committee.match(str(x)))).all():
            # Apply the transformation using a lambda function to create the hyperlink
            df[col] = df[col].apply(lambda y: f'[{y}](https://www.fec.gov/data/committee/{y})')
            
        if df[col].apply(lambda x: bool(pattern_candidate.match(str(x)))).all():
            # Apply the transformation using a lambda function to create the hyperlink
            df[col] = df[col].apply(lambda y: f'[{y}](https://www.fec.gov/data/candidate/{y})')
            
    df = df.apply(lambda col: col.map(format_decimal))
    return df

async def run_single_message(
    message: str,
    conversation_history = [],
    chainlit_msg_object = None,
    semantic_parser_class = None,
    save_to_local = None,
):
    step_counter = 1
    
    # only pass in chainlit callback if chainlit_msg_object is not None
    callbacks = []
        
    # this is a "placeholder" state in case the agent bypasses all actions
    # happens for instance when agent is chitchatting
    state = {
        "question": message,
        "conversation_history": conversation_history,
        "generated_sqls": [],
        "actions": [],
        "entity_linking_results": {}
    }
    
    # find out if there is an entity substitution dict from previous runs
    if conversation_history and "entity_linking_results" in conversation_history[-1]:
        entity_linking_results = conversation_history[-1]["entity_linking_results"]
    else:
        entity_linking_results = {}
    
    if semantic_parser_class is None:
        semantic_parser_class = cl.user_session.get("semantic_parser_class")
    
    async for chunk in semantic_parser_class.runnable.with_config(
        {"recursion_limit": 60, "max_concurrency": 50}
    ).astream_events(
        {
            "question": message,
            "conversation_history": conversation_history,
            "entity_linking_results": entity_linking_results
        },
        config=RunnableConfig(
            callbacks=callbacks,
            tags=[semantic_parser_class.database_name]
        ),
        version="v1"
    ):
        if chunk["name"] in [
            "execute_sql",
            "get_tables",
            "retrieve_tables_details",
            "search_graph",
            "entity_linking",
            "location_linking",
            "fetch_incoming_outgoing_edges",
            "execute_sparql",
        ] and chunk["event"] == "on_chain_end" \
        and chunk["tags"][0].startswith("seq:"):
            action = chunk["data"]["input"]["actions"][-1]
            if chainlit_msg_object:
                display_step_name = map_step_name_to_natural_names(action.action_name)
                print(f"Completed Step {step_counter}: {display_step_name}")
                step_counter += 1
            state = chunk["data"]["input"]
            
        elif chunk["name"] == "verify_domain_specific_instructions" \
            and chunk["event"] == "on_chain_end" \
            and chunk["tags"][0].startswith("seq:"):
            
            action = chunk["data"]["output"]["actions"][-1] if chunk["data"]["output"] and "actions" in chunk["data"]["output"] and chunk["data"]["output"]["actions"] else None
            if chainlit_msg_object and action:
                print("Reviewing Final SQL")
            state = chunk["data"]["input"]
        
    action_history = []
    domain_specific_instructions = set()
    # if actions not in state then something strange happend.
    if "actions" in state:
        actions = state["actions"]
        
        get_tables_schema_results = []
        for i, a in enumerate(actions):
            include_observation = True
            # we always include `get_tables_schema` results
            # but delete results from SQL queries in prior tunrs
            if i < len(actions) - 7 and a.action_name in [
                "execute_sql",
            ]:
                include_observation = False
            elif a.action_name == "stop":
                include_observation = False
                
            # exclude get_tables_schema results that are the same
            if a.action_name == "get_tables_schema":
                if a.observation in get_tables_schema_results:
                    include_observation = False
                else:
                    get_tables_schema_results.append(a.observation)
            
            action_history.append(a.to_jinja_string(include_observation))
        
        for i, a in enumerate(reversed(actions)):
            if a.action_name == "execute_sql":
                domain_specific_instructions = domain_specific_instructions.union(set(
                    retrieve_relevant_domain_specific_instructions(
                        a.action_argument,
                        state["domain_specific_instructions"],
                        controller_reporter="reporter"
                    )
                ))
    
    current_dir = os.path.dirname(__file__)
    instruction_path = os.path.join(current_dir, "kraken/prompts/conversation_reporter_instruction.prompt")
    input_template_path = os.path.join(current_dir, "kraken/prompts/conversation_reporter_input.prompt")

    with open(instruction_path, "r") as fd:
        reporter_llm_instruction = fd.read()

    with open(input_template_path, "r") as fd:
        reporter_llm_input_template = fd.read()
    
    reporter_llm_input = jinja2_formatter(
        reporter_llm_input_template,
        **{
            "question": state["question"],
            "conversation_history": state["conversation_history"],
            "action_history": action_history,
            "domain_specific_instructions": list(domain_specific_instructions)
        }
    )
    reporter_msgs = [
        {"role": "system", "content": reporter_llm_instruction},
        {"role": "user", "content": reporter_llm_input}
    ]
    

    stream = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4000,
        temperature=0.0,
        messages=reporter_msgs,
        response_format = { 
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": _ensure_strict_json_schema(SqlReporterResponse.model_json_schema(), path=()),
                "name": SqlReporterResponse.__name__,
            }
        },
        stream=True
    )
    
    whole_msg = ""
    for part in stream:
        if token := part.choices[0].delta.content or "":
            whole_msg += token
            if chainlit_msg_object:
                await chainlit_msg_object.stream_token(token)
    
    response = json.loads(whole_msg)
    report = response["report"]
    preprocessed_sql = response["generated_sql"]
    postprocessed_sql = response["generated_sql"]
    technical_explanation_of_SQL = response["technical_explanation_of_SQL"]
    simplified_explanation_of_SQL = response["simplified_explanation_of_SQL"]
    limitations = response["limitations"]
    suggested_next_turn_queries = response["suggested_next_turn_natural_language_queries"]
    
    msg_content, summary = "", ""
    if report is not None:
        msg_content = report
        summary = report
    
    result_count = -1
    sql_result = None
    if preprocessed_sql:
        substitution_dict = state["entity_linking_results"]
        postprocessed_sql = await postprocess_entities(postprocessed_sql, substitution_dict)

        sql_query_object = await DatatalkParser.sql_chain(
            postprocessed_sql,
            state
        )
        sql_result = sql_query_object.execution_result_sample
        
        if report is None:
            msg_content = "I attempted to answer your question by the following query\n\n"
            summary = ""
        
        if report is None or (report is not None and not sql_result):
            msg_content += f"```sql\n{postprocessed_sql}\n```\n\n"
            summary += f"```sql\n{preprocessed_sql}\n```\n\n" # in summary, we also just use preprocessed sql
            msg_content += f"### Simplified Explanation of SQL\n\n{simplified_explanation_of_SQL}\n\n"
            summary += f"### Simplified Explanation of SQL\n\n{simplified_explanation_of_SQL}\n\n"
            msg_content += f"### Technical Explanation of SQL\n\n{technical_explanation_of_SQL}\n\n"
            msg_content += f"### Limitations\n\n{limitations}\n\n"
            
            msg_content += f"## Result\n\n{sql_result}\n\n"
            summary += f"## Result\n\n{sql_result}\n\n"
        
        if sql_query_object.execution_result_full_dict is not None:
            result_count = len(sql_query_object.execution_result_full_dict)
    
    if chainlit_msg_object:
        chainlit_msg_object.content = msg_content
    
    if postprocessed_sql and chainlit_msg_object:
        json_file = cl.File(
            name=f"View Full Results, Edit, or Share",
            url="https://datatalk-sql.genie.stanford.edu/queries/new?sql=" + urllib.parse.quote(postprocessed_sql),
        )
        chainlit_msg_object.elements=[json_file]

    # only these two fields ("question" and "action_history") are relevant for future turns
    conversation_history.append(
        {
            "question": state["question"].strip(),
            "action_history": state["actions"],
            "entity_linking_results": state["entity_linking_results"] if preprocessed_sql else {},
            "response": msg_content.replace(postprocessed_sql, preprocessed_sql) if postprocessed_sql else postprocessed_sql # in conversation history, we only show preprocessed SQL
        }
    )
    
    if chainlit_msg_object:
        cl.user_session.set("conversation_history", conversation_history)
    
    if suggested_next_turn_queries:
        if chainlit_msg_object:
            chainlit_msg_object.actions = list(map(lambda x: cl.Action(name="follow-up-query", value=x, label=x), suggested_next_turn_queries))
        else:
            msg_content += "Suggested next turn queries: " + str(suggested_next_turn_queries)
    
    write_prompt_logs_to_file()
    
    class ActionEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Action):
                return obj.to_dict()
            return super().default(obj)
    
    conversation_history = json.dumps(conversation_history, cls=ActionEncoder)
    print(f"result count: {result_count}")
    res = {
        "agent_response": msg_content,
        "preprocessed_sql": preprocessed_sql,
        "generated_sql": postprocessed_sql,
        "conversation_history": conversation_history,
        "result_count": result_count,
        "summary": summary
    }
    
    if save_to_local and os.path.isdir(save_to_local):
        # Generate a random hash for the filename
        random_hash = hashlib.sha256(os.urandom(16)).hexdigest()
        file_path = os.path.join(save_to_local, f"{random_hash}.json")
        
        # Save the agent response to the file
        with open(file_path, "w") as file:
            json.dump({
                "agent_response": msg_content,
                "conversation_history": conversation_history,
                "sql_result": sql_result,
                "preprocessed_sql": preprocessed_sql,
                "generated_sql": postprocessed_sql,
            }, file, indent=2)
        os.chmod(file_path, 0o644)
        res["file_path"] = file_path
    
    print(f"total cost so far: {global_cost_counter.value}")
    return res