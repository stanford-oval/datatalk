import csv
import json
import operator
import re
from typing import Annotated, Optional, Sequence, TypedDict, Dict, Any, List

import pandas as pd
import chainlit as cl


from collections import defaultdict

from kraken.sql_utils import execute_sql


def json_to_panda_markdown(
    data: List[dict],
    head = 10,
    processing_fcn = None
) -> str:
    df = pd.DataFrame(data)
    
    # Determine the number of rows to show at the top and bottom
    num_head = head // 2
    num_tail = head - num_head
    
    # Create the truncated DataFrame
    if len(df) > head and head > 0:
        head_df = df.head(num_head)
        tail_df = df.tail(num_tail)
        
        # Number of omitted rows
        omitted_rows = len(df) - head
        
        # Create a custom row with '... (x omitted) | ... | ...'
        custom_row = pd.DataFrame({col: ['...'] for col in df.columns})
        custom_row[df.columns[0]] = f'... ({omitted_rows} omitted)'
        
        # Concatenate the head, custom row, and tail DataFrames
        truncated_df = pd.concat([head_df, custom_row, tail_df], ignore_index=True)
    else:
        truncated_df = df

    if processing_fcn:
        truncated_df = processing_fcn(truncated_df)

    return truncated_df.to_markdown(index=False)


def json_to_markdown_table(data):
    # Assuming the JSON data is a list of dictionaries
    # where each dictionary represents a row in the table
    if not data or not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("JSON data is not in the expected format for a table")

    def rearrange_list(lst):
        if "_id" in lst:
            lst.remove("_id")
            lst.insert(0, "_id")
        return lst

    # Extract headers, and put _id up front
    headers = rearrange_list(list(data[0].keys()))
    
    # Start building the Markdown table
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["-" * len(header) for header in headers]) + " |\n"

    # Add rows
    for row in data:
        markdown_table += "| " + " | ".join(str(row[header]) for header in headers) + " |\n"

    return markdown_table


def convert_json_to_table_format(data):
    # Load the JSON data
    if isinstance(data, str):
        data = json.loads(data)

    # Convert the JSON data to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Convert the DataFrame to a table format
    table = df.to_markdown(index=False)

    return table


class SqlQuery:
    def __init__(
        self,
        sql: Optional[str],
        table_w_ids: dict,
        database_name: str,
        embedding_server_address: str,
        source_file_mapping: dict,
        suql_model_name: str,
        db_type: str,
        db_secrets_file: str,
        suql_enabled: bool
    ):
        if "SELECT" not in sql:
            self.is_valid = False
        self.sql = SqlQuery.clean_sql(sql)
        self.table_w_ids = table_w_ids
        self.database_name = database_name
        self.embedding_server_address = embedding_server_address
        self.source_file_mapping = source_file_mapping
        self.is_valid = True
        self.result_count = -1

        self.execution_result_sample = None
        self.execution_result_full_dict = None
        self.execution_status = None
        
        self.suql_model_name = suql_model_name
        
        self.db_type = db_type
        self.db_secrets_file = db_secrets_file
        self.suql_enabled = suql_enabled
        # self.full_result_enums = full_result_enums

    @staticmethod
    def clean_sql(sql: str):
        if sql is None:
            return sql
        cleaned_sql = sql.strip()
        return cleaned_sql


    async def execute(
        self,
        processing_fcn = None
    ):
        if self.sql == "SELECT * FROM information_schema.tables WHERE table_schema = 'public';":
            self.execution_result_sample = "Prohibited. Use `get_tables_schema`"
            self.execution_result_full_dict = "Prohibited. Use `get_tables_schema`"
            return
        
        # TODO: probably need to perform some post processing by using column_names with self.execution_result
        execution_result, self.execution_status = await execute_sql(
            self.sql,
            self.table_w_ids,
            self.database_name,
            self.suql_model_name,
            self.embedding_server_address,
            self.source_file_mapping,
            self.db_type,
            self.db_secrets_file,
            self.suql_enabled
        )


        if execution_result is not None:
            # if self.sql in self.full_result_enums:
            #     self.execution_result_sample = json_to_panda_markdown(dict_result, head=-1)
            # else:
            
            head = 10
            # TODO: hardcoded
            if "SELECT * FROM transaction_type_codes" in self.sql:
                head = -1
            
            self.execution_result_sample = json_to_panda_markdown(
                execution_result,
                head = head,
                processing_fcn = processing_fcn
            )
            self.execution_result_full_dict = execution_result
            self.result_count = len(execution_result)

    def has_results(self) -> bool:
        return self.execution_result_sample is not None

    def __repr__(self):
        return f"Sql({self.sql})"

    def __hash__(self):
        return hash(self.sql)


def merge_dictionaries(dictionary_1: dict, dictionary_2: dict) -> dict:
    """
    Merges two dictionaries, combining their key-value pairs.
    If a key exists in both dictionaries, the value from dictionary_2 will overwrite the value from dictionary_1.

    Parameters:
        dictionary_1 (dict): The first dictionary.
        dictionary_2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary containing the merged key-value pairs.
    """
    merged_dict = dictionary_1.copy()  # Start with a copy of the first dictionary
    merged_dict.update(
        dictionary_2
    )  # Update with the second dictionary, overwriting any duplicates
    return merged_dict


def merge_sets(set_1: set, set_2: set) -> set:
    return set_1 | set_2


def add_item_to_list(_list: list, item) -> list:
    ret = _list.copy()
    # if item not in ret:
    ret.append(item)
    return ret


class BaseParserState(TypedDict):
    question: str
    engine: str
    generated_sqls: Annotated[list[SqlQuery], add_item_to_list]
    final_sql: SqlQuery
    action_counter: Annotated[int, operator.add]
    total_action_counter: Annotated[int, operator.add]
    examples: list[str]
    table_schemas: list[dict[str, str]]
    conversation_history: list
    domain_specific_instructions: dict
    response: str
    verify_domain_specific_instructions_counter: Annotated[int, operator.add]
    starter_cache: dict
    db_type: str
    suql_enabled: bool
    db_secrets_file: str
    database_name: str
    last_timer: float
    num_init_steps_cached: int
    available_actions: list
    table_w_ids: dict
    embedding_server_address: str
    source_file_mapping: str
    suql_model_name: str
    entity_linking_results: dict


class Action:
    possible_actions = [
        "get_tables",
        "retrieve_tables_details",
        "execute_sql",
        "get_examples",
        "entity_linking",
        "location_linking",
        "stop",
        "execute_sparql", # in graph
        "search_graph", # in graph
        "fetch_incoming_outgoing_edges", # in graph
        
        "error" # a special action to denote that an error occured in one of the action parsing
    ]

    # All actions have a single input parameter for now
    def __init__(self, thought: str, action_name: str, action_argument: str, observation: str = None):
        self.thought = thought
        self.action_name = action_name
        self.action_argument = action_argument
        self.observation = observation
        self.postprocessed_action_argument = action_argument  # for now - it may become different from standard action_argument if entity_linking gets called
        self.result_count = None # this only applies to execute SQLs.

        assert self.action_name in Action.possible_actions, f"{self.action_name} not in permitted actions"

    # Necessary for serializing action history for rule proposal generation
    def to_dict(self) -> Dict[str, Any]:
        """Convert Action to dictionary for JSON serialization"""
        return {
            "thought": self.thought,
            "action_name": self.action_name,
            "action_argument": self.action_argument,
            "observation": self.observation
        }
        
    @classmethod
    def from_dict(cls, data):
        """Dynamically instantiate an object from a dictionary."""
        return cls(**data)

    def to_jinja_string(self, include_observation: bool) -> str:
        if not self.observation:
            observation = "Did not find any results."
        else:
            observation = self.observation
        ret = f"Thought: {self.thought}\nAction: {self.action_name}({self.action_argument})\n"
        if include_observation:
            ret += f"Observation: {observation}\n"
        else:
            # for SQL queries, we still append a string showing how long the output was
            if self.result_count is not None:
                if self.result_count == -1:
                    ret += f"Did not find any results"
                else:
                    ret += f"Observation omitted due to length. Retrieved {self.result_count} rows\n"
            ret += f"Observation omitted due to length.\n"
            
        return ret
    
    async def print_chainlit(self, step_name):
        async with cl.Step(name=step_name, type="tool", show_input=False) as step_thought:
            thought = self.thought
            action = f"{self.postprocessed_action_argument}"
            
            if not self.action_argument:
                 step_thought.output = f"""### Thought
{thought}
"""

            elif not self.observation:
                step_thought.output = f"""### Thought
{thought}

### Action
```sql\n{action}
```
"""
                
            elif self.action_name in ['entity_linking', 'location_linking']:
                observation = self.observation
                observation="```python\n"+str(observation)+"\n```"
                
                step_thought.output = f"""### Thought
{thought}

### Action
```python\n{action}
```

### Observation
{observation}
"""

            else:
                observation = self.observation
                if self.action_name == "get_tables_schema":
                    observation="```python\n"+str(observation)+"\n```"
            
                step_thought.output = f"""### Thought
{thought}

### Action
```sql\n{action}
```

### Observation
{observation}
"""
        

    def __repr__(self) -> str:
        if not self.observation:
            observation = "Did not find any results."
        else:
            observation = self.observation
        return f"Thought: {self.thought}\nAction: {self.action_name}({self.action_argument})\nObservation: {observation}"

    def __eq__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return (
            self.action_name == other.action_name
            and self.action_argument == other.action_argument
        )

    def __hash__(self):
        return hash((self.action_name, self.action_argument))


class DatatalkParserState(BaseParserState):
    actions: Sequence[Action]

def compute_domain_specific_instructions(csv_file_path):
    if not csv_file_path:
        return dict()
    
    res = defaultdict(lambda: {
        "reporter": [],
        "controller": [],
    })
    with open(csv_file_path, "r") as fd:
        reader = csv.DictReader(fd)
    
        for row in reader:
            # there are 2 formats for this file for now.
            
            # One using simple 3 row headers: table_name, instruction, report_controller_flag
            # Another one from the rule-proposing framework.
            if "trigger_condition" in row.keys():
                table_name_row_name = "trigger_condition"
            else:
                table_name_row_name = "table_name"
            
            if "report_controller_flag" not in row or row["report_controller_flag"] == "0":
                res[row[table_name_row_name]]["reporter"].append(row["instruction"])
                res[row[table_name_row_name]]["controller"].append(row["instruction"])
            elif row["report_controller_flag"] == "1":
                res[row[table_name_row_name]]["reporter"].append(row["instruction"])
            elif row["report_controller_flag"] == "2":
                res[row[table_name_row_name]]["controller"].append(row["instruction"])
            else:
                raise ValueError()
    return res