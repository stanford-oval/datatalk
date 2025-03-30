import mysql.connector
import pandas as pd
from typing import List, Tuple
import json
import argparse
from dotenv import load_dotenv
import os


treat_as_enum = {
    'stanford_api_data': ['disorder_type', 'event_type', 'sub_event_type', 'region', 'inter1', 'inter2', 'interaction', 'country', 'civilian_targeting']
} # User pre-defines {table_name : [col_name_1, col_name_2, ...]} for all column names they want to pre-process as ENUMs


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

def get_table_names(cursor) -> List[str]:
    cursor.execute("SHOW TABLES")
    return [table[0] for table in cursor.fetchall()]

# We don't have persmissions for SHOW CREATE TABLE command, so manually generating the command using metadata from information_schema DB
# Includes column comments even though ACLED API data has no comments for any columns
def generate_create_table(cursor, table_name):
    column_query = f"""
    SELECT column_name, data_type, column_type, column_comment
    FROM information_schema.columns 
    WHERE table_name = '{table_name}';
    """
    cursor.execute(column_query)
    columns = cursor.fetchall()

    pk_query = f"""
    SELECT column_name
    FROM information_schema.key_column_usage
    WHERE table_name = '{table_name}'
    AND constraint_name = 'PRIMARY'
    ORDER BY ordinal_position;
    """
    cursor.execute(pk_query)
    pk_columns = [row[0] for row in cursor.fetchall()]

    # Start building the CREATE TABLE command
    create_table_command = f"CREATE TABLE `{table_name}` ("

    column_definitions = []
    for column in columns:
        column_name, data_type, column_type, column_comment = column
        
        definition = f"  `{column_name}` {column_type}"
        
        if column_name in pk_columns:
            definition += " PRIMARY KEY"
        
        if column_comment:
            definition += f" COMMENT '{column_comment}'"
        
        column_definitions.append(definition)

    create_table_command += ",".join(column_definitions)

    if len(pk_columns) > 1:
        pk_constraint = f",  PRIMARY KEY (`{'`, `'.join(pk_columns)}`)"
        create_table_command += pk_constraint

    create_table_command += ");"

    print(f"Generated a create_table command of: ", create_table_command)
    return create_table_command

def get_table_columns(cursor, table_name: str) -> List[Tuple[str, str]]:
    cursor.execute(f"DESCRIBE `{table_name}`")
    return [(row[0], row[1]) for row in cursor.fetchall()]

def get_primary_key(cursor, table_name: str) -> str:
    cursor.execute(f"SHOW KEYS FROM `{table_name}` WHERE Key_name = 'PRIMARY'")
    result = cursor.fetchone()
    return result[4] if result else None

# Returns all text-like columns, but we can manually change this based on what we want to count as 'free-text'
def get_free_text_fields(columns: List[Tuple[str, str]]) -> str:
    ft_fields = [col[0] for col in columns if col[1].lower().startswith('text') or col[1].lower().startswith('varchar')]
    #print('These are the ft_fields: ', ft_fields)
    return ";".join(ft_fields)

def generate_create_enum(cursor, table_name, col_name):
    print(f"Generating an ENUM row for table_name {table_name} and col_name of {col_name}")
    distinct_query = f"SELECT DISTINCT {col_name} FROM {table_name}"
    cursor.execute(distinct_query)
    results = cursor.fetchall()
    distinct_values = [str(result[0]) for result in results]
    distinct_values_str = ', '.join(distinct_values)

    create_enum_cmd = f"CREATE TYPE {col_name} AS ENUM ({distinct_values_str})"
    print(f'Generated create ENUM command of: {create_enum_cmd}')
    return create_enum_cmd



def create_lookup_table(connection) -> pd.DataFrame:
    cursor = connection.cursor()
    
    lookup_data = []
    
    for table_name in get_table_names(cursor):
        create_command = generate_create_table(cursor, table_name)
        columns = get_table_columns(cursor, table_name)
        id_field_name = get_primary_key(cursor, table_name)
        free_text_fields = get_free_text_fields(columns)
        
        # Add main table entry
        lookup_data.append({
            "type": "table",
            "table_name": table_name,
            "id_field_name": id_field_name,
            "table_CREATE_command": create_command,
            "free_text_fields": free_text_fields,
            "other_description": None
        })
    
    for table_name in treat_as_enum:
        for col_name in treat_as_enum[table_name]:
            create_command = generate_create_enum(cursor, table_name, col_name)

            lookup_data.append({
                "type": "enum",
                "table_name": table_name,
                "id_field_name": None,
                "table_CREATE_command": create_command,
                "free_text_fields": None,
                "other_description": None
            })
         

    cursor.close()
    return pd.DataFrame(lookup_data)

def main():
    parser = argparse.ArgumentParser(description="Create a MySQL lookup table and save it to a CSV file.")
    parser.add_argument('--lookup-dest-path', type=str, required=True,
                        help='The destination path for the lookup table CSV file.')
    parser.add_argument('--env-file', type=str, required=True,
                        help='Path to the .env file containing database credentials.')

    args = parser.parse_args()

    # Load environment variables from the specified .env file
    load_dotenv(args.env_file)

    # Retrieve database credentials from environment variables
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    ssl_ca = os.getenv('SSL_CA')
    database = os.getenv('DATABASE')

    # Ensure all required environment variables are set
    required_vars = ['DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'SSL_CA', 'DATABASE']
    missing_vars = [var for var in required_vars if os.getenv(var) is None]

    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    connection = connect_to_mysql(host, port, user, password, database, ssl_ca)
    lookup_df = create_lookup_table(connection)
    
    # Save the lookup table to a CSV file
    lookup_df.to_csv(args.lookup_dest_path, index=False)
    print(f"Lookup table saved to {args.lookup_dest_path}")
    
    connection.close()

if __name__ == "__main__":
    main()