from sqlalchemy import create_engine, text
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import pandas as pd
from config import DB_CONFIG
Base = declarative_base() 


def get_connection_string():
    return( f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
           f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    
def get_engine():
    connection_string = get_connection_string()
    engine = create_engine(connection_string)
    return engine


def create_tables():
    """Create all tables in database"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Tables created successfully!")
    
def save_to_db(dataframe, table_name, if_exists='replace'):
    #we add some exception functions here for error troubleshooting
    try:
        engine = get_engine()
        dataframe.to_sql(table_name, con=engine, if_exists=if_exists, 
                        index=True, chunksize=1000)
        print(f"Successfully saved {len(dataframe)} rows to table '{table_name}'")
    except Exception as e:
        print(f"Error saving to database: {e}")
 
def read_from_db(table_name):    
    try:
        engine = get_engine()
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, con=engine)
        print(f"Loaded {len(df)} rows from '{table_name}")
        return df
    
    except Exception as e:
        print(f"Loaded {len(df)} rows from '{table_name}'")
        return None

def execute_query(query):
    try:
        engine = get_engine()
        result  = pd.read_sql(query, con=engine)
        return result 
    
    except Exception as e :
        print("Error executing query, check for any typos here")
        return None
    
    #the beauty here is we are using the query word passed and in the result variable we ue this query variable and we extract/query from the database that is present (WOWWW!)

def query_with_filter(table_name, where_clause = None):
    engine = get_engine()
    #see the magic/beauty here 
    if where_clause:
        query = f"SELECT * FROM {table_name} WHERE {where_clause}" # see how we actually queruied stuff here
    else:
        query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, con=engine)

'''We use this data in so good manner and a more clean manner in the 
Tabletestfile.py file, this is just sooo mindblowingly good


example usage : 

save_to_db(agg, 'customer_insights', if_exists='replace')
    
    return agg

# Run the analysis
result = intelligence()

# Read back from MySQL
loaded_data = read_from_db('customer_insights')
print(loaded_data)

# Query with filters
high_value = execute_query("""
    SELECT * FROM customer_insights 
    WHERE monetary > (SELECT AVG(monetary) FROM customer_insights)
    ORDER BY monetary DESC
    LIMIT 10
""")
print(high_value)

# Also save the cleaned data
save_to_db(a, 'raw_transactions', if_exists='replace')
'''

'''
Complete workflow administation here

1. Load CSV
   ↓
2. Clean & Transform Data
   ↓
3. Aggregate & Create Insights (DataFrame)
   ↓
4. Save to MySQL using to_sql()
   ↓
5. Query from MySQL using read_sql()
   ↓
6. Analyze Results


'''