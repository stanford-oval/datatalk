import re

def oracle_to_postgres_type(oracle_type):
    oracle_type = oracle_type.upper()
    type_mapping = {
        'NUMBER': 'NUMERIC',
        'VARCHAR2': 'VARCHAR',
        # Add other mappings as needed
    }

    # Handle special cases with precision and scale
    match = re.match(r'(\w+)\s*\((\d+)(?:,(\d+))?\)', oracle_type)
    if match:
        base_type = match.group(1)
        precision = match.group(2)
        scale = match.group(3)

        # Map the base type
        if base_type == 'NUMBER':
            if scale:
                return f'NUMERIC({precision},{scale})'
            else:
                return f'NUMERIC({precision})'
        elif base_type == 'VARCHAR2':
            return f'VARCHAR({precision})'
        else:
            # Return the base type if it's not explicitly handled
            return f'{type_mapping.get(base_type, base_type)}({precision}{"," + scale if scale else ""})'
    
    # all DATE(...) types are mapped to just DATE in Postgres
    elif oracle_type.startswith('DATE('):
        return 'DATE'
    
    # Handle types without precision/scale, or unrecognized formats
    return type_mapping.get(oracle_type, oracle_type)

if __name__ == "__main__":
    oracle_type_1 = 'NUMBER(14,2)'
    oracle_type_2 = 'VARCHAR2 (9)'
    oracle_type_3 = 'DATE(MM/DD/YYYY)'

    postgres_type_1 = oracle_to_postgres_type(oracle_type_1)
    postgres_type_2 = oracle_to_postgres_type(oracle_type_2)
    postgres_type_3 = oracle_to_postgres_type(oracle_type_3)

    print(postgres_type_1)
    print(postgres_type_2)
    print(postgres_type_3)