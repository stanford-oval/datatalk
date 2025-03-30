import psycopg2
import csv
import pandas as pd
from tqdm import tqdm
import json
import os
import subprocess
from typing import List, Tuple, Dict
import re
from datetime import datetime
from suql.postgresql_connection import execute_sql as execute_sql_direct
from ingestion_tools import oracle_to_postgres_type
from io import StringIO
import stat


FORMATS = ['percentage', 'float', 'int', 'date', 'time', 'timestamp', 'interval', 'units']


class FormatDetection:
    def __init__(self, is_empty, original_val):
        self.is_empty = is_empty
        self.original_val = original_val
        self.format_possibility = {}
        self.format_conversion = {}
        self.possible_formats = set()
        self.format_fn = {
            'percentage': self.detect_percentage,
            'int': self.detect_int,
            'float': self.detect_float,
            'date': self.detect_date,
            'time': self.detect_time,
            'timestamp': self.detect_timestamp,
            'interval': self.detect_interval,
            'units': self.detect_units,
        }
        self.clean_val = self.clean(original_val)

    def set_format(self, format, possibility, conversion):
        self.format_possibility[format] = possibility
        self.format_conversion[format] = conversion

    def get_format(self, format):
        return self.format_possibility[format], self.format_conversion[format]

    def build_possible_formats(self):
        for format in FORMATS:
            if self.get_format(format)[0]:
                self.possible_formats.add(format)

    @staticmethod
    def clean(val):
        if type(val) == str:
            # Remove all parentheses
            pattern = r'\([^)]*\)'
            val = re.sub(pattern, '', val)
            # Keep only useful characters
            pattern = r'[^0-9a-zA-Z:. %/+-]'
            val = re.sub(pattern, '', val)
            val = val.strip()
        return val

    @staticmethod
    def detect_percentage(value):
        if type(value) == str:
            match = re.fullmatch(r'-?\d+(\.\d+)?%', value)
            if match:
                number_without_percent_sign = match.group(0)[:-1]
                conversion = float(number_without_percent_sign) / 100
                return {'possibility': True, 'conversion': conversion}
        return {'possibility': False, 'conversion': None}

    @staticmethod
    def detect_int(value):
        try:
            conversion = int(value)
            if -2147483648 < conversion < 2147483647:
                return {'possibility': True, 'conversion': conversion}
        except:
            pass
        return {'possibility': False, 'conversion': None}

    @staticmethod
    def detect_float(value):
        try:
            conversion = float(value)
            return {'possibility': True, 'conversion': conversion}
        except:
            return {'possibility': False, 'conversion': None}
        
    @staticmethod
    def detect_date(
        value,
        assert_must_match=False
    ):
        if type(value) == str:
            # note difference between "%B %d  %Y" and "%B %d %Y": the former can result from a removed comma (so double spaces in between)
            formats = [
                "%Y-%m-%d",         # ISO 8601 format
                "%d %B %Y",         # e.g., 25 December 2023
                "%B %d %Y",         # e.g., December 25 2023
                "%d %B",            # e.g., 25 December (assumes current year)
                "%B %d",            # e.g., December 25 (assumes current year)
                "%B %Y",            # e.g., December 2023
                "%d/%m/%Y",         # e.g., 25/12/2023
                "%m/%d/%Y",         # e.g., 12/25/2023
                "%d-%m-%Y",         # e.g., 25-12-2023
                "%m-%d-%Y",         # e.g., 12-25-2023
                "%Y/%m/%d",         # e.g., 2023/12/25
            ]
            
            for format in formats:
                try:
                    conversion = datetime.strptime(value, format).strftime('%Y-%m-%d')
                    return {'possibility': True, 'conversion': conversion}
                except ValueError:
                    pass
        
        if assert_must_match:
            raise ValueError(f"detect_date: Value {value} cannot be matched")
        return {'possibility': False, 'conversion': None}

    @staticmethod
    def detect_time(value):
        if type(value) == str:
            match = re.fullmatch(r'^\d+:\d+(\.\d+)?$', value)
            if match:
                conversion = match.group(0)
                return {'possibility': True, 'conversion': conversion}
        return {'possibility': False, 'conversion': None}

    @staticmethod
    def detect_interval(value):
        if type(value) == str:
            match = re.fullmatch(r'^\d+:\d+(\.\d+)?$', value)
            if match:
                conversion = match.group(0)
                if int(conversion.split(':')[0]) > 23:
                    return {'possibility': True, 'conversion': conversion}
        return {'possibility': False, 'conversion': None}

    @staticmethod
    def detect_timestamp(value):
        if type(value) == str:
            try:
                # Check if the string contains fractional seconds
                if '.' in value:
                    date_time_str, nanoseconds = value.split('.')
                    # Truncate nanoseconds to microseconds if needed
                    if len(nanoseconds) > 6:
                        nanoseconds = nanoseconds[:6]
                else:
                    date_time_str, nanoseconds = value, '0'  # No fractional seconds
                
                # Parse the main part of the timestamp
                conversion = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
                # Add the microseconds to the conversion
                conversion = conversion.replace(microsecond=int(nanoseconds))
                
                return {'possibility': True, 'conversion': conversion.strftime('%Y-%m-%d %H:%M:%S.%f')}
            except ValueError:
                return {'possibility': False, 'conversion': None}
        return {'possibility': False, 'conversion': None}

    @staticmethod
    def detect_units(value):
        # if type(value) == str:
        #     match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', value)
        #     if match:
        #         conversion = float(match.group(1))
        #         return {'possibility': True, 'conversion': conversion}
        return {'possibility': False, 'conversion': None}


class ColumnDetection():
    def __init__(self, all_detection: List[FormatDetection]):
        self.all_detection = all_detection

    def all_empty(self):
        return all([det.is_empty for det in self.all_detection])

    def all_text(self):
        return all([det.possible_formats == set() for det in self.all_detection])

    def get_original_column(self):
        return [det.original_val for det in self.all_detection]

    def get_nonempty_detections(self):
        return [det for det in self.all_detection if not det.is_empty]

    def get_nonempty_mask(self):
        return [not det.is_empty for det in self.all_detection]

    def get_formats(self):
        format_set = []
        all_formats = []
        nonempty_formats = []
        for det in self.all_detection:
            cell_formats = [format for format, possible in det.format_possibility.items() if possible]
            all_formats.append(cell_formats)
            if not det.is_empty:
                nonempty_formats.append(cell_formats)
            format_set += cell_formats
        return all_formats, nonempty_formats, set(format_set)

    def get_conversion(self, format=None, format_list=None, format_priority=None):
        updated_column = []
        if format:
            for det in self.all_detection:
                if det.is_empty:
                    updated_column.append('')
                else:
                    updated_column.append(det.format_conversion[format])
            return updated_column

        elif format_list:
            for format, det in zip(format_list, self.all_detection):
                if det.is_empty:
                    updated_column.append('')
                elif format is None:
                    updated_column.append(det.original_val)
                else:
                    updated_column.append(det.format_conversion[format])
            return updated_column

        elif format_priority:
            for det in self.all_detection:
                if det.is_empty:
                    updated_column.append('')
                    continue
                found_format = False
                for format in format_priority:
                    possible, conversion = det.get_format(format)
                    if possible == True:
                        updated_column.append(conversion)
                        found_format = True
                        break
                if not found_format:
                    raise Exception(f'Cannot convert format "{format}" for the detection: ', det)
            return updated_column


def detect_column(column: List[str]) -> ColumnDetection:
    res = []
    for item in column:
        # if empty
        if item == '':
            res.append(FormatDetection(is_empty=True, original_val=''))
            continue
        
        # otherwise, detect
        detection = FormatDetection(is_empty=False, original_val=item)
        for format in FORMATS:
            detection.set_format(format, **detection.format_fn[format](detection.clean_val))
        detection.build_possible_formats()
        res.append(detection)

    res = ColumnDetection(res)
    return res


def aggregate_detection(col_detection: ColumnDetection) -> Tuple[str, Dict, List[str]]:
    
    # TEXT case 1: all cells are empty or no formats detected for all cells
    # psql type: TEXT
    if col_detection.all_empty() or col_detection.all_text():
        return 'TEXT', {}, col_detection.get_original_column()
    
    # init
    all_detected_formats, nonempty_formats, format_set = col_detection.get_formats()
    nonempty_mask = col_detection.get_nonempty_mask()

    # INT: all non-empty ones has int
    # psql type: INT
    if all('int' in _ for _ in nonempty_formats):
        conversion = col_detection.get_conversion(format='int')
        mapping = dict(zip(conversion, col_detection.get_original_column()))
        return 'INT', mapping, conversion

    # DATE: all non-empty ones are date
    # psql type: DATE
    if all('date' in _ for _ in nonempty_formats):
        conversion = col_detection.get_conversion(format='date')
        mapping = dict(zip(conversion, col_detection.get_original_column()))
        return 'DATE', mapping, conversion

    # INTERVAL: all non-empty ones are time, and has one or more interval
    # This has to be before the TIME logics
    # psql type: INTERVAL
    if any('interval' in _ for _ in nonempty_formats) and all('time' in _ for _ in nonempty_formats):
    # if format_set == set(['time', 'interval']) or format_set == set(['interval']):
        conversion = col_detection.get_conversion(format_priority=['interval', 'time'])
        mapping = dict(zip(conversion, col_detection.get_original_column()))
        return 'INTERVAL', mapping, conversion

    # TIME: all non-empty ones are time, no interval
    # psql type: TIME
    if format_set == all('time' in _ for _ in nonempty_formats):
        conversion = col_detection.get_conversion(format='time')
        mapping = dict(zip(conversion, col_detection.get_original_column()))
        return 'TIME', mapping, conversion

    # TIMESTAMP: all non-empty ones are timestamp
    # psql type: TIMESTAMP
    if all('timestamp' in _ for _ in nonempty_formats):
        conversion = col_detection.get_conversion(format='timestamp')
        mapping = dict(zip(conversion, col_detection.get_original_column()))
        return 'TIMESTAMP', mapping, conversion

    # UNITS: all non-empty ones are units
    # psql type: FLOAT
    if format_set == all('units' in _ for _ in nonempty_formats):
        conversion = col_detection.get_conversion(format='units')
        mapping = dict(zip(conversion, col_detection.get_original_column()))
        return 'FLOAT', mapping, conversion

    # FLOAT: all non-empty ones are not date/time/interval/text, can be a mix of float/percentage/int/units
    format_list = []
    float_flag = True

    for idx, cell_formats in enumerate(all_detected_formats):
        if len(cell_formats) == 0:
            if nonempty_mask[idx]:
                # text cell, break
                float_flag = False
                break
            format_list.append(None)
        else:
            if any([_ in cell_formats for _ in ['date', 'time', 'interval']]):
                float_flag = False
                break
            elif not any([_ in cell_formats for _ in ['float', 'percentage', 'int', 'units']]):
                float_flag = False
                break
            for format in ['float', 'percentage', 'int', 'units']:
                if format in cell_formats:
                    format_list.append(format)
                    break
    if float_flag:
        conversion = col_detection.get_conversion(format_list=format_list)
        mapping = dict(zip(conversion, col_detection.get_original_column()))
        return 'FLOAT', mapping, conversion

    # TEXT case 2: everything else
    return 'TEXT', {}, col_detection.get_original_column()

def get_unique_values_pd_df(column):
    if isinstance(column.iloc[0], list):
        # Flatten the list of lists and get unique values
        unique_values = set(value for sublist in column for value in sublist)
    else:
        # Get unique values directly
        unique_values = set(column.unique())
    return unique_values

def process_csv(
    csv_filepath_or_dict,
    database,
    delimiter=',',
    quotechar='"',
):
    """
    The header file should be a CSV file that contains the following (useful) fields:
    - Column name
    - Description (in FEC csvs, this is the concat of Field name and Description)
    - Data type
    
    `delimiter`, `quotechar` refer to the CSV structure, they are passed to the pandas `read_csv` function.
    """
    new_entries = [] # for recording into the `_lookup_table.csv`
    
    assert type(csv_filepath_or_dict) is dict and "csv_filepath" in csv_filepath_or_dict
    csv_filepath = csv_filepath_or_dict["csv_filepath"]
    table_name = os.path.splitext(os.path.basename(csv_filepath))[0]
       
    pre_declared_enums = {}
    free_text_fields = ""
       
    if "csv_filepath_header" not in csv_filepath_or_dict or not csv_filepath_or_dict["csv_filepath_header"]:
        df = pd.read_csv(csv_filepath, quotechar=quotechar, delimiter=delimiter, keep_default_na=False)
        id_field_name = csv_filepath_or_dict["id_field_name"] if "id_field_name" in csv_filepath_or_dict else None
        df, ordered_headers, _ = create_df_from_csv(df, id_field_name=id_field_name)
        create_table_param = ', '.join([f'"{header_name}" {header_type}' for header_name, header_type in ordered_headers])
        create_table_cmd = f"CREATE TABLE \"{table_name}\" ({create_table_param})"
        create_table_command = create_table_for_df(
            df,
            table_name, 
            ordered_headers,
            database,
            create_table_cmd
        )
        id_field_name = list(filter(lambda x: "PRIMARY KEY" in x[1], ordered_headers))[0][0]
        
        # write down the list of free text fields
        free_text_fields = [i[0] for i in ordered_headers if i[1] == "TEXT"]
        free_text_fields = ";".join(free_text_fields)
        
        new_entries.append({
            "type": "table",
            "table_name": table_name,
            "id_field_name": id_field_name,
            "table_CREATE_command": create_table_command,
            "free_text_fields": free_text_fields,
            "other_description": None
        })
        
        return pd.DataFrame(new_entries)
        
    else:
        assert "csv_filepath" in csv_filepath_or_dict and "csv_filepath_header" in csv_filepath_or_dict
        
        # "csv_filepath" field can also be a linux command to run to get the data
        if "command_to_execute" in csv_filepath_or_dict:
            try:
                subprocess.run(csv_filepath_or_dict["command_to_execute"], shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Command execution failed with error:")
                raise e
        
        # create a CREATE command based on what is inside the header field
        df = pd.read_csv(csv_filepath_or_dict["csv_filepath_header"])
        if "Description" in df:
            df["Description"] = df["Description"].fillna('')
        create_table_command = f'CREATE TABLE "{table_name}" (\n'
        
        # if a description exists in this dict, include it in the create table commmand
        if "description" in csv_filepath_or_dict:
            description = '\n'.join([f"--{i}" for i in csv_filepath_or_dict["description"].split('\n')])
            create_table_command = description + "\n" + create_table_command
        
        ordered_headers = []
        
        # determine if the CSV has a primary key
        no_primary_key = True
        for _, row in df.iterrows():
            if "PRIMARY KEY" in row["Data type"]:
                no_primary_key = False
        
        for index, row in df.iterrows():
            column_name = row["Column name"]
            column_datatype = oracle_to_postgres_type(row["Data type"])
            ordered_headers.append((column_name, column_datatype))
            if no_primary_key and index == 0:
                column_datatype += " PRIMARY KEY"
            
            column_description = ""
            if "Field name" in row and "Description" in row and row["Field name"] and row["Description"]:
                column_description += row["Field name"] + ". " + row["Description"]
            elif "Description" in row:
                column_description += row["Description"]
            elif "Field name" in row:
                column_description += row["Field name"]
                
            column_description = column_description.replace("\n", ", ").replace("\r", "") # replace "\n" in comments since it will cause problems
            
            # standardize to type `DATE` (special modifier info is already stored in `ordered_headers`)
            if column_datatype.strip().startswith("DATE"):
                column_datatype = "DATE"
                
            # for field declared as ENUM, first assume an ENUM type - the exact type will be declared later
            if column_datatype.strip() == "ENUM":
                column_datatype = f"{table_name}_{column_name}_enum"
                pre_declared_enums[column_datatype] = column_name
            elif ''.join(column_datatype.strip()) == "ENUM[]":
                column_datatype = f"{table_name}_{column_name}_enum"
                pre_declared_enums[column_datatype] = column_name
                # the type should actually be of list
                column_datatype = f"{table_name}_{column_name}_enum" + "[]"
                
            if index == df.index[-1]:
                create_table_command += f""""{column_name}" {column_datatype}"""
            else:
                create_table_command += f""""{column_name}" {column_datatype},"""
            if column_description:
                create_table_command += f" -- {column_description}"
            create_table_command += "\n"
            
        
        create_table_command += ");"
        id_field_name = df.iloc[0]["Column name"]

    # prepare converters to convert columns of certain types to their canonical representation
    converters = {}
    # for string types, inform pandas that they should be kept as strings (do not cast to numbers)
    dtype = {}
    for col_name, col_type in ordered_headers:
        # "DATE(MM/DD/YYYY)" is automatically handled by datetime
        if re.sub(r'\s+', '', col_type) == "DATE(MMDDYYYY)":
            converters[col_name] = lambda x: datetime.strptime(x, '%m%d%Y').strftime('%Y-%m-%d') if x else None
        if col_type.startswith("VARCHAR"):
            dtype[col_name] = "string"
        if ''.join(col_type.strip()) == "ENUM[]":
            from ast import literal_eval
            converters[col_name] = literal_eval

    
    data_path = csv_filepath
    if "special_processing_fcn" in csv_filepath_or_dict:
        if type(csv_filepath_or_dict["special_processing_fcn"]) is str:
            special_processing_fcn = eval(csv_filepath_or_dict["special_processing_fcn"])
        else:
            special_processing_fcn = csv_filepath_or_dict["special_processing_fcn"]
    else:
        special_processing_fcn = lambda x:x
        
    # now, if there are pre-declared ENUM values, find them
    create_table_command_sql = create_table_command # for ENUM declarations, we do not include them in the look up table
    
    
    # if pre_declared_enums is present, then we have no choice but to read in the whole file
    if pre_declared_enums:
        # if df constructed directly from csv, then no need to read from csv again
        if "special_processing_fcn" in csv_filepath_or_dict:
            with open(csv_filepath, 'r') as file:
                data = special_processing_fcn(file.read())
                data_path = StringIO(data)
        
        if not("csv_filepath_header" not in csv_filepath_or_dict or not csv_filepath_or_dict["csv_filepath_header"]):
            df = pd.read_csv(
                data_path,
                delimiter=delimiter,
                header=None,
                names=list(map(lambda x:x[0], ordered_headers)),
                quotechar=quotechar,
                converters=converters,
                dtype=dtype,
                keep_default_na=False,
                index_col=False
            )
            
        df = df.where(pd.notnull(df), None)
        
        for column_datatype, column_name in pre_declared_enums.items():
            unique_values = get_unique_values_pd_df(df[column_name])
            
            def replace_single_strs_then_wrap(x):
                x = x.replace("'", "''")
                return "'" + x + "'"
            
            unique_values_stmt = ', '.join(list(map(replace_single_strs_then_wrap, unique_values)))
            create_type_command = f"CREATE TYPE {column_datatype} AS ENUM ({unique_values_stmt});\n"
            create_table_command_sql = f"DROP TYPE IF EXISTS {column_datatype};\n" + create_type_command + create_table_command
            
            new_entries.append({
                "type": "enum",
                "table_name": column_datatype,
                "id_field_name": None,
                "table_CREATE_command": create_type_command,
                "free_text_fields": None,
                "other_description": None
            })
        
        create_table_for_df(
            df,
            table_name, 
            ordered_headers,
            database,
            create_table_cmd=create_table_command_sql # use the command with type declaration
        )
    else:
        # for col_name, col_type in ordered_headers:
        #     # "DATE(MM/DD/YYYY)" is automatically handled by datetime
        #     if re.sub(r'\s+', '', col_type) == "DATE(MMDDYYYY)":
        #         converters[col_name] = lambda x: datetime.strptime(x, '%m%d%Y').date() if x else None
        chunk_size = 10000
        execute_sql_direct(f'DROP TABLE IF EXISTS "{table_name}"', database, user="creator_role", password="creator_role", commit_in_lieu_fetch=True)
        execute_sql_direct(create_table_command_sql, database, user="creator_role", password="creator_role", commit_in_lieu_fetch=True)
        
        # with open(csv_filepath, 'r') as file:
        #     data = special_processing_fcn(file.read())
        #     data_path = StringIO(data)
        conn = psycopg2.connect(
            database=database,
            user="creator_role",
            password="creator_role",
            host="127.0.0.1",
            port="5432",
            options="-c client_encoding=UTF8",
        )
        cur = conn.cursor()
        
        def count_lines_with_wc(file_path):
            result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
            return int(result.stdout.split()[0])
        # def bad_line_handler(bad_line):
        #     print(f"Bad line encountered: {bad_line}")
        #     raise ValueError
        
        with tqdm(total=count_lines_with_wc(data_path)) as pbar:
            with open(data_path, 'r') as file:  
                while True:
                    lines = []
                    try:
                        # Try to read up to chunk_size lines from the file
                        for _ in range(chunk_size):
                            lines.append(next(file))
                    except StopIteration:
                        pass
                    
                    if not lines:
                        break  # Exit the loop if there are no more lines to process
                    
                    # Apply special processing function to the chunk
                    processed_data = special_processing_fcn("".join(lines))
                    data_buffer = StringIO(processed_data)
                    columns = list(map(lambda x:x[0], ordered_headers))
                    
                    if not quotechar:
                        chunk = pd.read_csv(
                            data_buffer,
                            delimiter=delimiter,
                            header=None,
                            names=columns,
                            quoting=csv.QUOTE_NONE,
                            converters=converters,
                            dtype=dtype,
                            keep_default_na=False,
                            index_col=False,
                        )
                    else:
                        chunk = pd.read_csv(
                            data_buffer,
                            delimiter=delimiter,
                            header=None,
                            names=columns,
                            quotechar=quotechar,
                            converters=converters,
                            dtype=dtype,
                            keep_default_na=False,
                            index_col=False,
                        )

                    
                    for col_name, col_type in ordered_headers:
                        if col_type.strip().endswith("[]"):
                            # turn the list into a string
                            # the string is simply {val1,val2} (no quotes)
                            # the only thing that needs to be escaped are the commas, which are escaped with "\\\\"
                            df[col_name] = df[col_name].apply(lambda x: '{' + ','.join(item.replace(",", "\\\\,") for item in (x)) + '}')
                    
                    # chunk['TRANSACTION_DT'] = pd.to_datetime(chunk['TRANSACTION_DT'])
                    # print(chunk['TRANSACTION_DT'].dtype)
                    # pd.set_option('display.max_columns', None)
                    # print(chunk)
                    # print(set(chunk['TRANSACTION_DT'].apply(type)))


                    # Copy the processed chunk into PostgreSQL using pgcopy
                    # with conn.cursor() as cursor:
                        # mgr = CopyManager(conn, table_name, columns)
                        # mgr.copy(chunk)
                    output = StringIO()
                    if quotechar:
                        chunk.to_csv(output, sep='|', quoting=csv.QUOTE_MINIMAL, header=False, index=False, escapechar='\\')
                    else:
                        chunk.to_csv(output, sep='|', quoting=csv.QUOTE_NONE, header=False, index=False, escapechar='\\')
                    output.seek(0)
                    cur.copy_from(output, table_name, sep='|', null='')
                    
                    pbar.update(len(lines))
        
        cur.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO select_user;")
        conn.commit()
        conn.close()
        
    
    new_entries.append({
        "type": "table",
        "table_name": table_name,
        "id_field_name": id_field_name,
        "table_CREATE_command": create_table_command,
        "free_text_fields": free_text_fields,
        "other_description": None
    })
    return pd.DataFrame(new_entries)

def ensure_unique_id(
    df,
    ordered_headers,
    id_field_name
):
    if id_field_name is None:
        id_field_name = "id"
    
    if id_field_name in df.columns and df[id_field_name].is_unique:
        id_field_index = next((i for i, item in enumerate(ordered_headers) if item[0] == id_field_name), -1)
        ordered_headers[id_field_index] = (id_field_name, f"{ordered_headers[id_field_index][1]} PRIMARY KEY")
            
    else:
        # Create '_id' column with auto-incrementing index if 'id' does not exist
        df['_id'] = range(1, len(df) + 1)
        ordered_headers.insert(0, ("_id", "INT PRIMARY KEY"))
    
    return df, ordered_headers

def create_df_from_csv(
    df,
    id_field_name=None
):
    header = df.columns.tolist()

    # creates the cells
    cells = df.to_dict(orient='list')

    # go through columns and process
    ordered_headers = []
    column_mapping = {}
    clean_cells = {}
    
    for header, column in cells.items():
        type_name, current_column_mapping, converted_column = detect_convert_column(column)
        ordered_headers.append((header, type_name))
        column_mapping.update({header: current_column_mapping})
        clean_cells[header] = converted_column
    
    # construct the df
    df = pd.DataFrame(clean_cells)
    df, ordered_headers = ensure_unique_id(
        df,
        ordered_headers,
        id_field_name
    )
    
    df = df[[_[0] for _ in ordered_headers]]  # keep only the columns used

    return df, ordered_headers, column_mapping


def detect_convert_column(column):
    # this function returns three values: 
    #   1) the converted type of the column, 
    #   2) the mapping within the column
    #   3) the converted column, with updated values

    detection_result = detect_column(column)
    aggregated_result = aggregate_detection(detection_result)
    converted_type, mapping, converted_column = aggregated_result

    return converted_type, mapping, converted_column


def create_table_for_df(
    df, 
    table_name,
    ordered_headers,
    database,
    create_table_cmd
):
    pd.set_option('display.max_columns', None)

    for col_name, col_type in ordered_headers:
        if col_type.strip().endswith("[]"):
            # turn the list into a string
            # the string is simply {val1,val2} (no quotes)
            # the only thing that needs to be escaped are the commas, which are escaped with "\\\\"
            df[col_name] = df[col_name].apply(lambda x: '{' + ','.join(item.replace(",", "\\\\,") for item in (x)) + '}')
    
    execute_sql_direct(f'DROP TABLE IF EXISTS "{table_name}"', database, user="creator_role", password="creator_role", commit_in_lieu_fetch=True)

    execute_sql_direct(create_table_cmd, database, user="creator_role", password="creator_role", commit_in_lieu_fetch=True)

    conn = psycopg2.connect(
        database=database,
        user="creator_role",
        password="creator_role",
        host="127.0.0.1",
        port="5432",
        options="-c client_encoding=UTF8",
    )
    cur = conn.cursor()
    
    output = StringIO()
    df.to_csv(output, sep='|', quoting=csv.QUOTE_MINIMAL, header=False, index=False)
    output.seek(0)
    
    cur.copy_from(output, table_name, sep='|', null='')
    cur.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO select_user;")
    
    conn.commit()
    cur.close()
    conn.close()
    
    return create_table_cmd


def process_all_csvs(
    all_csv_file_paths,
    database,
    lookup_table_path = None,
    create_db = False
):
    """
    Ingests all CSV files as specified in the list of `all_csv_file_paths`, specified either as a list or a local file path.
    
    Create a lookup CSV path at `lookup_table_path` if specified, or at the same directory as the first CSV file.
    """
    if create_db:
        from ingestion_createdb import create_newdb
        create_newdb(database=database, use_psql_user=True)
        print(f"created postgres database name = {database}")
    
    if not lookup_table_path:
        if type(all_csv_file_paths[0]) is str:
            lookup_table_path = os.path.join(os.path.dirname(all_csv_file_paths[0]), "_lookup_table.csv")
        else:
            lookup_table_path = os.path.join(os.path.dirname(all_csv_file_paths[0]["csv_filepath"]), "_lookup_table.csv")
    
    res = []
    for i in all_csv_file_paths:
        i_res = process_csv(
            i, 
            database,
            delimiter=i["delimiter"] if "delimiter" in i else ",",
            quotechar=i["quotechar"] if "quotechar" in i else '"',
        )
        res.append(i_res)
        filepath = i["csv_filepath"]
        print (f"Processed {filepath}: {str(i_res)}")
    
    result_df = pd.concat(res, ignore_index=True)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(lookup_table_path), exist_ok=True)

    # Set read, write, and execute permissions for the directory for other users
    os.chmod(os.path.dirname(lookup_table_path), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # Create the CSV file using touch
    touch_file_path = os.path.join(os.path.dirname(lookup_table_path), "domain_specific_instructions.csv")
    subprocess.run(['touch', touch_file_path])

    # Set read and write permissions for the CSV file for other users
    os.chmod(touch_file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)

    # Save the DataFrame to a CSV file
    result_df.to_csv(lookup_table_path, index=False)

    # Set read and write permissions for the CSV file created by `to_csv`
    os.chmod(lookup_table_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
    
if __name__ == "__main__": 
    declarations = []
    declarations += [
        {
            "csv_filepath": f"/data1/insight-bench/data/notebooks/csvs/flag-{i}.csv",
        }
        for i in range(1, 11)
    ]
    
    process_all_csvs(
        declarations,
        "insight_bench",
        lookup_table_path="/data1/datatalk_domains/insight_bench/_lookup_table.csv"
    )