import subprocess
import argparse

def create_newdb(
    database,
    use_psql_user = False
):
    if use_psql_user:
        create_newdb_under_psql(database)
        return
    
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    con = psycopg2.connect(
        dbname='postgres',
        user='creator_role',
        password='creator_role',
        host='localhost',
    )

    con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cur = con.cursor()

    # Use the psycopg2.sql module instead of string concatenation 
    # in order to avoid sql injection attacks.
    cur.execute(sql.SQL("CREATE DATABASE {}").format(
        sql.Identifier(database))
    )
    
def create_newdb_under_psql(database):
    # Log in as the postgres user and create the database
    subprocess.run(['sudo', 'su', '-', 'postgres', '-c', f'createdb {database}'])

    # List of SQL commands to execute one by one
    sql_commands = [
        "CREATE ROLE select_user WITH PASSWORD 'select_user';",
        "GRANT SELECT ON ALL TABLES IN SCHEMA public TO select_user;",
        "ALTER ROLE select_user LOGIN;",
        f"CREATE ROLE creator_role WITH PASSWORD 'creator_role';",
        f'GRANT CREATE ON DATABASE "{database}" TO creator_role;',
        'GRANT CREATE ON SCHEMA public TO creator_role;',
        'ALTER ROLE creator_role LOGIN;'
    ]

    for cmd in sql_commands:
        subprocess.run([
            'sudo', 'su', '-', 'postgres', '-c', f'psql {database} -c "{cmd}"'
        ])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a new PostgreSQL database.')
    parser.add_argument('database', type=str, help='The name of the database to create.')
    args = parser.parse_args()

    create_newdb(args.database)