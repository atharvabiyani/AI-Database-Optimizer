# -*- coding: utf-8 -*-
"""FINAL_connection_with_RAG_and_LLM.py

Local version of the RAG and LLM implementation.
"""

import pandas as pd
import snowflake.connector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
from openai import OpenAI
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def setup_connections():
    print("Setting up Snowflake connection...")
    snowflake_config = {
        "user": "LOBSTER",
        "password": "GTG59NQNK5HYm3x",
        "account": "SFEDU02-IOB07152",
        "warehouse": "MY_WAREHOUSE_INSRDB",
        "database": "INSURANCE_DB_10K",
        "schema": "PUBLIC"  # Add default schema
    }

    try:
        conn = snowflake.connector.connect(
            user=snowflake_config["user"],
            password=snowflake_config["password"],
            account=snowflake_config["account"],
            warehouse=snowflake_config["warehouse"],
            database=snowflake_config["database"],
            schema=snowflake_config["schema"]  # Add schema to connection
        )
        print("Successfully connected to Snowflake")
        
        # Test the connection and get database info
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
        db_name, schema_name = cursor.fetchone()
        print(f"Connected to database: {db_name}, schema: {schema_name}")
        
        # List available schemas
        cursor.execute("SHOW SCHEMAS")
        schemas = cursor.fetchall()
        print("\nAvailable schemas:")
        for schema in schemas:
            print(f"- {schema[1]}")
        
        # Get table count
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
        """)
        table_count = cursor.fetchone()[0]
        print(f"\nFound {table_count} tables in schema {schema_name}")
        
        if table_count == 0:
            print("\nWARNING: No tables found in the current schema. Please check:")
            print("1. The schema name is correct")
            print("2. You have access to the schema")
            print("3. The schema contains tables")
            
            # Try to list tables in the schema
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            if tables:
                print("\nTables found using SHOW TABLES:")
                for table in tables:
                    print(f"- {table[1]}")
        
        cursor.close()
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        raise

    print("\nSetting up embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="query_optimization",
        embedding_function=embeddings,
        persist_directory="./chroma_query_db"
    )
    print("Vector store initialized")

    return conn, vector_store

def query_snowflake(sql):
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(result, columns=columns)
        else:
            df = pd.DataFrame()

        cursor.close()
        return df
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

def store_user_query(query_text):
    query_id = str(uuid.uuid4())

    vector_store.add_texts(
        texts=[query_text],
        metadatas=[{"query_id": query_id, "type": "original_query"}],
        ids=[query_id]
    )

    return query_id

def extract_and_store_database_metadata(database_name="INSURANCE_DB_10K"):
    query_snowflake(f"USE DATABASE {database_name}")
    metadata = {
        "database": database_name,
        "schemas": [],
        "tables": [],
        "views": [],
        "columns": [],
        "relationships": []
    }

    schemas_query = "SHOW SCHEMAS"
    schemas_df = query_snowflake(schemas_query)

    user_schemas = schemas_df[~schemas_df['name'].isin(['INFORMATION_SCHEMA'])]

    for _, schema_row in user_schemas.iterrows():
        schema_name = schema_row['name']
        metadata["schemas"].append({"name": schema_name})

        query_snowflake(f"USE SCHEMA {schema_name}")

        tables_query = "SHOW TABLES"
        tables_df = query_snowflake(tables_query)

        for _, table_row in tables_df.iterrows():
            table_name = table_row['name']
            table_info = {
                "name": table_name,
                "schema": schema_name,
                "kind": table_row.get('kind', 'TABLE'),
                "columns": []
            }

            columns_query = f"""
            SELECT
                column_name,
                data_type,
                character_maximum_length,
                is_nullable,
                column_default
            FROM
                information_schema.columns
            WHERE
                table_schema = '{schema_name}'
                AND table_name = '{table_name}'
            ORDER BY
                ordinal_position
            """
            columns_df = query_snowflake(columns_query)

            for _, col_row in columns_df.iterrows():
                column_info = {
                    "name": col_row['COLUMN_NAME'],
                    "data_type": col_row['DATA_TYPE'],
                    "max_length": col_row['CHARACTER_MAXIMUM_LENGTH'],
                    "nullable": col_row['IS_NULLABLE'] == 'YES',
                    "default": col_row['COLUMN_DEFAULT']
                }
                table_info["columns"].append(column_info)

                metadata["columns"].append({
                    **column_info,
                    "table": table_name,
                    "schema": schema_name
                })

            table_info["inferred_primary_key"] = []
            id_columns = [col["name"] for col in table_info["columns"]
                         if col["name"].lower().endswith('_id') or
                            col["name"].lower() == 'id' or
                            col["name"].lower().endswith('key')]
            if id_columns:
                table_info["inferred_primary_key"] = id_columns

            potential_fks = []
            for col in table_info["columns"]:
                col_name = col["name"].lower()

                if "_id" in col_name and col_name != "id" and not col_name.endswith("_rid"):
                    ref_table = col_name.split('_id')[0]
                    potential_fks.append({
                        "column": col["name"],
                        "potential_referenced_table": ref_table,
                        "confidence": "inferred"
                    })

            if potential_fks:
                table_info["inferred_foreign_keys"] = potential_fks

                for fk in potential_fks:
                    metadata["relationships"].append({
                        "type": "inferred_foreign_key",
                        "source_schema": schema_name,
                        "source_table": table_name,
                        "source_column": fk["column"],
                        "target_table": fk["potential_referenced_table"],
                        "confidence": "inferred"
                    })

            try:
                row_count_query = f"SELECT COUNT(*) as row_count FROM {schema_name}.{table_name}"
                row_count_df = query_snowflake(row_count_query)
                if not row_count_df.empty:
                    table_info["row_count"] = int(row_count_df.iloc[0]['ROW_COUNT'])
            except Exception as e:
                table_info["row_count"] = 0

            metadata["tables"].append(table_info)

        views_query = "SHOW VIEWS"
        views_df = query_snowflake(views_query)

        for _, view_row in views_df.iterrows():
            view_name = view_row['name']
            view_info = {
                "name": view_name,
                "schema": schema_name
            }

            try:
                view_def_query = f"SELECT GET_DDL('VIEW', '{schema_name}.{view_name}')"
                view_def_df = query_snowflake(view_def_query)

                if not view_def_df.empty:
                    view_info["definition"] = view_def_df.iloc[0][0]
            except Exception as e:
                view_info["definition"] = ""

            metadata["views"].append(view_info)

    # Store metadata in vector store
    db_doc_text = f"Database: {metadata['database']}\n\n"
    db_doc_text += f"Contains {len(metadata['schemas'])} schemas, {len(metadata['tables'])} tables, "
    db_doc_text += f"{len(metadata['views'])} views, and {len(metadata['relationships'])} inferred relationships."

    vector_store.add_texts(
        texts=[db_doc_text],
        metadatas=[{"type": "database_info", "database": metadata["database"]}],
        ids=[f"db_{metadata['database']}"]
    )

    for table in metadata["tables"]:
        table_doc_text = f"Table: {table['schema']}.{table['name']}\n\n"

        table_doc_text += "Columns:\n"
        for col in table["columns"]:
            table_doc_text += f"- {col['name']} ({col['data_type']})"
            if col.get("nullable") == False:
                table_doc_text += " NOT NULL"
            table_doc_text += "\n"

        if "inferred_primary_key" in table and table["inferred_primary_key"]:
            table_doc_text += f"\nInferred Primary Key: {', '.join(table['inferred_primary_key'])}\n"

        if "inferred_foreign_keys" in table and table["inferred_foreign_keys"]:
            table_doc_text += "\nInferred Foreign Keys:\n"
            for fk in table["inferred_foreign_keys"]:
                table_doc_text += f"- {fk['column']} may reference table {fk['potential_referenced_table']}\n"

        if "row_count" in table:
            table_doc_text += f"\nApproximate Row Count: {table['row_count']:,}\n"

        vector_store.add_texts(
            texts=[table_doc_text],
            metadatas=[{
                "type": "table_info",
                "database": metadata["database"],
                "schema": table["schema"],
                "table": table["name"]
            }],
            ids=[f"table_{metadata['database']}_{table['schema']}_{table['name']}"]
        )

    for view in metadata["views"]:
        view_doc_text = f"View: {view['schema']}.{view['name']}\n\n"

        if "definition" in view:
            view_doc_text += f"Definition:\n{view['definition']}\n"

        vector_store.add_texts(
            texts=[view_doc_text],
            metadatas=[{
                "type": "view_info",
                "database": metadata["database"],
                "schema": view["schema"],
                "view": view["name"]
            }],
            ids=[f"view_{metadata['database']}_{view['schema']}_{view['name']}"]
        )

    for relation in metadata["relationships"]:
        rel_doc_text = f"Relationship: {relation['source_schema']}.{relation['source_table']}.{relation['source_column']} → "
        rel_doc_text += f"{relation['target_table']}\n\n"
        rel_doc_text += f"Type: {relation['type']} (Confidence: {relation['confidence']})\n"

        vector_store.add_texts(
            texts=[rel_doc_text],
            metadatas=[{
                "type": "relationship_info",
                "database": metadata["database"],
                "source_schema": relation["source_schema"],
                "source_table": relation["source_table"],
                "target_table": relation["target_table"]
            }],
            ids=[f"rel_{metadata['database']}_{relation['source_schema']}_{relation['source_table']}_{relation['source_column']}"]
        )

    return {
        "database": database_name,
        "schema_count": len(metadata["schemas"]),
        "table_count": len(metadata["tables"]),
        "view_count": len(metadata["views"]),
        "column_count": len(metadata["columns"]),
        "relationship_count": len(metadata["relationships"])
    }

def identify_and_store_redundant_objects(database_name="INSURANCE_DB_10K"):
    print(f"Analyzing database {database_name} for redundant objects...")

    tables_docs = vector_store.similarity_search_with_score(
        query="Table information",
        k=1000,
        filter={
            "$and": [
                {"type": {"$eq": "table_info"}},
                {"database": {"$eq": database_name}}
            ]
        }
    )

    views_docs = vector_store.similarity_search_with_score(
        query="View information",
        k=1000,
        filter={
            "$and": [
                {"type": {"$eq": "view_info"}},
                {"database": {"$eq": database_name}}
            ]
        }
    )

    print(f"Found {len(tables_docs)} tables and {len(views_docs)} views to analyze")

    tables = []
    for doc, _ in tables_docs:
        table_content = doc.page_content
        table_name = table_content.split("\n")[0].replace("Table: ", "").strip()

        columns = []
        column_types = {}
        if "Columns:" in table_content:
            columns_section = table_content.split("Columns:")[1]
            if "\n\n" in columns_section:
                columns_section = columns_section.split("\n\n")[0]

            for line in columns_section.split("\n"):
                if line.strip().startswith("- "):
                    col_line = line.strip()[2:].strip()
                    col_name = col_line.split(" (")[0].strip().lower()

                    if "(" in col_line and ")" in col_line:
                        data_type = col_line.split("(")[1].split(")")[0].strip().lower()
                        column_types[col_name] = data_type

                    columns.append(col_name)

        primary_keys = []
        if "Primary Key:" in table_content:
            pk_section = table_content.split("Primary Key:")[1]
            if "\n" in pk_section:
                pk_section = pk_section.split("\n")[0]
            primary_keys = [pk.strip().lower() for pk in pk_section.split(",")]
        elif "Inferred Primary Key:" in table_content:
            pk_section = table_content.split("Inferred Primary Key:")[1]
            if "\n" in pk_section:
                pk_section = pk_section.split("\n")[0]
            primary_keys = [pk.strip().lower() for pk in pk_section.split(",")]

        row_count = 0
        if "Approximate Row Count:" in table_content:
            count_section = table_content.split("Approximate Row Count:")[1]
            if "\n" in count_section:
                count_section = count_section.split("\n")[0]
            try:
                row_count = int(count_section.strip().replace(",", ""))
            except ValueError:
                pass

        schema_name = ""
        if "." in table_name:
            schema_name, table_name = table_name.split(".")

        tables.append({
            "full_name": table_content.split("\n")[0].replace("Table: ", "").strip(),
            "schema": schema_name,
            "name": table_name,
            "columns": columns,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "row_count": row_count,
            "content": table_content
        })

    views = []
    for doc, _ in views_docs:
        view_content = doc.page_content
        view_name = view_content.split("\n")[0].replace("View: ", "").strip()

        definition = ""
        if "Definition:" in view_content:
            definition = view_content.split("Definition:")[1].strip()

        referenced_tables = []
        if definition:
            definition_lower = definition.lower()

            from_parts = definition_lower.split(" from ")
            if len(from_parts) > 1:
                for part in from_parts[1:]:
                    table_part = part
                    for keyword in [" where ", " join ", " group by ", " order by ", " having "]:
                        if keyword in table_part:
                            table_part = table_part.split(keyword)[0]

                    for table_ref in table_part.split(","):
                        table_ref = table_ref.strip()

                        if " as " in table_ref:
                            table_ref = table_ref.split(" as ")[0].strip()

                        elif " " in table_ref:
                            table_ref = table_ref.split(" ")[0].strip()

                        table_ref = table_ref.strip("()));")

                        if table_ref and not table_ref.startswith("select"):
                            referenced_tables.append(table_ref)

            join_parts = definition_lower.split(" join ")
            if len(join_parts) > 1:
                for part in join_parts[1:]:
                    table_ref = part.split(" ")[0].strip()

                    table_ref = table_ref.strip("()));")

                    if table_ref and not table_ref.startswith("select"):
                        referenced_tables.append(table_ref)

        schema_name = ""
        local_view_name = view_name
        if "." in view_name:
            schema_name, local_view_name = view_name.split(".")

        views.append({
            "full_name": view_name,
            "schema": schema_name,
            "name": local_view_name,
            "definition": definition,
            "referenced_tables": referenced_tables,
            "content": view_content
        })

    redundant_tables = []

    for i, table1 in enumerate(tables):
        for table2 in tables[i+1:]:
            if table1["schema"] != table2["schema"]:
                continue

            common_columns = set(table1["columns"]) & set(table2["columns"])
            total_columns = set(table1["columns"]) | set(table2["columns"])

            if not common_columns:
                continue

            overlap_percentage = len(common_columns) / len(total_columns) * 100

            if overlap_percentage >= 70:
                redundancy = {
                    "table1": table1["full_name"],
                    "table2": table2["full_name"],
                    "overlap_percentage": overlap_percentage,
                    "common_columns": sorted(list(common_columns)),
                    "unique_to_table1": sorted(list(set(table1["columns"]) - set(table2["columns"]))),
                    "unique_to_table2": sorted(list(set(table2["columns"]) - set(table1["columns"]))),
                    "row_counts": {
                        table1["full_name"]: table1["row_count"],
                        table2["full_name"]: table2["row_count"]
                    }
                }

                if table1["primary_keys"] and table2["primary_keys"]:
                    pk_overlap = set(table1["primary_keys"]) & set(table2["primary_keys"])
                    if pk_overlap:
                        redundancy["common_primary_keys"] = sorted(list(pk_overlap))

                redundant_tables.append(redundancy)

    naming_patterns = {}

    for table in tables:
        name = table["name"].lower()

        for pattern in ["tmp_", "temp_", "_backup", "_old", "_new", "_archive", "_history"]:
            if pattern in name:
                base_name = name.replace(pattern, "")
                if base_name not in naming_patterns:
                    naming_patterns[base_name] = []
                naming_patterns[base_name].append({
                    "full_name": table["full_name"],
                    "pattern": pattern,
                    "columns": table["columns"],
                    "row_count": table["row_count"]
                })

    naming_redundancies = {base: tables for base, tables in naming_patterns.items() if len(tables) > 1}

    redundant_views = []

    for i, view1 in enumerate(views):
        for view2 in views[i+1:]:
            if view1["schema"] != view2["schema"]:
                continue

            common_refs = set(view1["referenced_tables"]) & set(view2["referenced_tables"])

            if not common_refs:
                continue

            from difflib import SequenceMatcher

            def similarity(a, b):
                return SequenceMatcher(None, a, b).ratio()

            definition_similarity = 0
            if view1["definition"] and view2["definition"]:
                definition_similarity = similarity(
                    view1["definition"].lower(),
                    view2["definition"].lower()
                ) * 100

            if definition_similarity >= 60 or (len(common_refs) / max(len(view1["referenced_tables"]), len(view2["referenced_tables"])) >= 0.8):
                redundancy = {
                    "view1": view1["full_name"],
                    "view2": view2["full_name"],
                    "definition_similarity": definition_similarity,
                    "common_referenced_tables": sorted(list(common_refs)),
                    "unique_to_view1": sorted(list(set(view1["referenced_tables"]) - set(view2["referenced_tables"]))),
                    "unique_to_view2": sorted(list(set(view2["referenced_tables"]) - set(view1["referenced_tables"])))
                }
                redundant_views.append(redundancy)

    table_duplicating_views = []

    for view in views:
        view_def = view["definition"].lower() if view["definition"] else ""

        if view_def.startswith("select") and " from " in view_def:
            from_part = view_def.split(" from ")[1]

            has_complex_logic = any(keyword in from_part for keyword in
                                   [" join ", " where ", " group by ", " having ", " order by ", " limit "])

            if not has_complex_logic and "," not in from_part:
                table_name = from_part.strip().split(" ")[0].strip()

                matching_tables = [t for t in tables if t["full_name"].lower() == table_name or t["name"].lower() == table_name]

                if matching_tables:
                    matching_table = matching_tables[0]
                    table_duplicating_views.append({
                        "view": view["full_name"],
                        "duplicates_table": matching_table["full_name"],
                        "view_definition": view["definition"]
                    })

    report = f"# Database Structure Analysis: Redundancy Report\n\n"
    report += f"## Overview\n\n"
    report += f"Database: {database_name}\n"
    report += f"Total Tables Analyzed: {len(tables)}\n"
    report += f"Total Views Analyzed: {len(views)}\n\n"

    if redundant_tables:
        report += f"## Potentially Redundant Tables ({len(redundant_tables)} pairs)\n\n"
        for i, redundancy in enumerate(redundant_tables, 1):
            report += f"### {i}. {redundancy['table1']} and {redundancy['table2']}\n\n"
            report += f"**Overlap:** {redundancy['overlap_percentage']:.1f}% of columns match\n\n"

            report += f"**Common Columns ({len(redundancy['common_columns'])}):**\n"
            for col in redundancy['common_columns'][:10]:
                report += f"- {col}\n"
            if len(redundancy['common_columns']) > 10:
                report += f"- ... and {len(redundancy['common_columns']) - 10} more\n"

            if redundancy['unique_to_table1']:
                report += f"\n**Columns Unique to {redundancy['table1']}:**\n"
                for col in redundancy['unique_to_table1'][:5]:
                    report += f"- {col}\n"
                if len(redundancy['unique_to_table1']) > 5:
                    report += f"- ... and {len(redundancy['unique_to_table1']) - 5} more\n"

            if redundancy['unique_to_table2']:
                report += f"\n**Columns Unique to {redundancy['table2']}:**\n"
                for col in redundancy['unique_to_table2'][:5]:
                    report += f"- {col}\n"
                if len(redundancy['unique_to_table2']) > 5:
                    report += f"- ... and {len(redundancy['unique_to_table2']) - 5} more\n"

            if 'common_primary_keys' in redundancy:
                report += f"\n**Common Primary Keys:** {', '.join(redundancy['common_primary_keys'])}\n"

            report += f"\n**Row Counts:**\n"
            report += f"- {redundancy['table1']}: {redundancy['row_counts'][redundancy['table1']]:,}\n"
            report += f"- {redundancy['table2']}: {redundancy['row_counts'][redundancy['table2']]:,}\n"

            report += f"\n**Recommendation:** Consider merging these tables or clarifying their distinct purposes.\n\n"
            report += "---\n\n"

    if naming_redundancies:
        report += f"## Tables with Suspicious Naming Patterns ({len(naming_redundancies)} groups)\n\n"
        for base_name, related_tables in naming_redundancies.items():
            report += f"### Base name: '{base_name}'\n\n"
            report += f"**Related Tables:**\n"

            for table in related_tables:
                report += f"- {table['full_name']} ({table['pattern']} pattern, {table['row_count']:,} rows)\n"

            report += f"\n**Recommendation:** Verify if these are temporary/backup tables and consider cleaning up redundant ones.\n\n"
            report += "---\n\n"

    if redundant_views:
        report += f"## Potentially Redundant Views ({len(redundant_views)} pairs)\n\n"
        for i, redundancy in enumerate(redundant_views, 1):
            report += f"### {i}. {redundancy['view1']} and {redundancy['view2']}\n\n"
            report += f"**Definition Similarity:** {redundancy['definition_similarity']:.1f}%\n\n"

            report += f"**Common Referenced Tables:**\n"
            for table in redundancy['common_referenced_tables']:
                report += f"- {table}\n"

            if redundancy['unique_to_view1']:
                report += f"\n**Tables Referenced Only by {redundancy['view1']}:**\n"
                for table in redundancy['unique_to_view1']:
                    report += f"- {table}\n"

            if redundancy['unique_to_view2']:
                report += f"\n**Tables Referenced Only by {redundancy['view2']}:**\n"
                for table in redundancy['unique_to_view2']:
                    report += f"- {table}\n"

            report += f"\n**Recommendation:** Review these view definitions and consider consolidating them.\n\n"
            report += "---\n\n"

    if table_duplicating_views:
        report += f"## Views Duplicating Tables ({len(table_duplicating_views)})\n\n"
        for i, duplication in enumerate(table_duplicating_views, 1):
            report += f"### {i}. {duplication['view']} duplicates {duplication['duplicates_table']}\n\n"
            report += f"**View Definition:**\n```sql\n{duplication['view_definition']}\n```\n\n"
            report += f"**Recommendation:** Unless this view adds security or abstraction value, consider using the table directly.\n\n"
            report += "---\n\n"

    if not (redundant_tables or naming_redundancies or redundant_views or table_duplicating_views):
        report += "## No Significant Redundancies Found\n\n"
        report += "The analysis did not identify any significant structural redundancies in this database.\n\n"

    consolidation_recommendations = []

    for redundancy in redundant_tables:
        if redundancy['overlap_percentage'] >= 90:
            table_to_keep = redundancy['table1']
            table_to_remove = redundancy['table2']

            if redundancy['row_counts'][redundancy['table2']] > redundancy['row_counts'][redundancy['table1']]:
                table_to_keep = redundancy['table2']
                table_to_remove = redundancy['table1']

            recommendation = {
                "type": "merge_tables",
                "description": f"Merge highly redundant tables",
                "objects": [table_to_remove, table_to_keep],
                "action": f"Merge {table_to_remove} into {table_to_keep}",
                "sql": f"-- Add missing columns to {table_to_keep}\n"
            }

            if table_to_keep == redundancy['table1']:
                missing_columns = redundancy['unique_to_table2']
            else:
                missing_columns = redundancy['unique_to_table1']

            for col in missing_columns:
                recommendation["sql"] += f"ALTER TABLE {table_to_keep} ADD COLUMN {col} VARCHAR; -- Set appropriate data type\n"

            recommendation["sql"] += f"\n-- Insert data from {table_to_remove} that doesn't already exist in {table_to_keep}\n"
            recommendation["sql"] += f"-- Customize the INSERT statement based on your primary key and data requirements\n"

            consolidation_recommendations.append(recommendation)

    for duplication in table_duplicating_views:
        recommendation = {
            "type": "remove_duplicating_view",
            "description": f"Remove view that duplicates a table",
            "objects": [duplication['view']],
            "action": f"Drop {duplication['view']} and use {duplication['duplicates_table']} directly",
            "sql": f"DROP VIEW {duplication['view']};"
        }
        consolidation_recommendations.append(recommendation)

    for base_name, related_tables in naming_redundancies.items():
        sorted_tables = sorted(related_tables, key=lambda x: x['row_count'], reverse=True)

        if len(sorted_tables) > 1:
            table_to_keep = sorted_tables[0]['full_name']
            tables_to_remove = [t['full_name'] for t in sorted_tables[1:]]

            recommendation = {
                "type": "cleanup_temp_tables",
                "description": f"Clean up temporary/backup tables for '{base_name}'",
                "objects": [table_to_keep] + tables_to_remove,
                "action": f"Keep {table_to_keep} and remove {len(tables_to_remove)} related tables",
                "sql": "-- Verify these are truly redundant before executing!\n"
            }

            for table in tables_to_remove:
                recommendation["sql"] += f"DROP TABLE {table};\n"

            consolidation_recommendations.append(recommendation)

    if consolidation_recommendations:
        report += f"## Consolidation Recommendations\n\n"
        report += "The following SQL statements can help consolidate redundant objects:\n\n"

        for i, rec in enumerate(consolidation_recommendations, 1):
            report += f"### {i}. {rec['description']}\n\n"
            report += f"**Action:** {rec['action']}\n\n"
            report += f"**SQL:**\n```sql\n{rec['sql']}\n```\n\n"
            report += "---\n\n"

    vector_store.add_texts(
        texts=[report],
        metadatas=[{"type": "redundancy_report", "database": database_name}],
        ids=[f"redundancy_report_{database_name}_{uuid.uuid4()}"]
    )

    return {
        "redundant_tables": redundant_tables,
        "naming_redundancies": naming_redundancies,
        "redundant_views": redundant_views,
        "table_duplicating_views": table_duplicating_views,
        "consolidation_recommendations": consolidation_recommendations,
        "report": report
    }

def analyze_query_patterns_for_rag(database_name="INSURANCE_DB_10K", num_queries_to_analyze=20):
    tables_docs = vector_store.similarity_search_with_score(
        query="Table information",
        k=1000,
        filter={
            "$and": [
                {"type": {"$eq": "table_info"}},
                {"database": {"$eq": database_name}}
            ]
        }
    )

    try:
        query_sql = f"""
        SELECT query_id, query_text, timestamp
        FROM QUERY_REPOSITORY
        ORDER BY timestamp DESC
        LIMIT {num_queries_to_analyze}
        """
        stored_queries_df = query_snowflake(query_sql)
    except Exception as e:
        stored_queries_df = pd.DataFrame(columns=["query_id", "query_text", "timestamp"])

    tables = {}
    for doc, _ in tables_docs:
        table_content = doc.page_content
        table_name = table_content.split("\n")[0].replace("Table: ", "").strip()

        schema_name = ""
        local_table_name = table_name
        if "." in table_name:
            schema_name, local_table_name = table_name.split(".")

        columns = []
        if "Columns:" in table_content:
            columns_section = table_content.split("Columns:")[1]
            if "\n\n" in columns_section:
                columns_section = columns_section.split("\n\n")[0]

            for line in columns_section.split("\n"):
                if line.strip().startswith("- "):
                    col_line = line.strip()[2:].strip()
                    col_name = col_line.split(" (")[0].strip()

                    data_type = ""
                    if "(" in col_line and ")" in col_line:
                        data_type = col_line.split("(")[1].split(")")[0].strip()

                    columns.append({"name": col_name, "type": data_type})

        primary_keys = []
        if "Primary Key:" in table_content:
            pk_section = table_content.split("Primary Key:")[1]
            if "\n" in pk_section:
                pk_section = pk_section.split("\n")[0]
            primary_keys = [pk.strip() for pk in pk_section.split(",")]
        elif "Inferred Primary Key:" in table_content:
            pk_section = table_content.split("Inferred Primary Key:")[1]
            if "\n" in pk_section:
                pk_section = pk_section.split("\n")[0]
            primary_keys = [pk.strip() for pk in pk_section.split(",")]

        row_count = 0
        if "Approximate Row Count:" in table_content:
            count_section = table_content.split("Approximate Row Count:")[1]
            if "\n" in count_section:
                count_section = count_section.split("\n")[0]
            try:
                row_count = int(count_section.strip().replace(",", ""))
            except ValueError:
                pass

        tables[table_name] = {
            "name": local_table_name,
            "schema": schema_name,
            "columns": columns,
            "primary_keys": primary_keys,
            "row_count": row_count
        }

    query_categories = {
        "simple_select": [],
        "joins": [],
        "aggregations": [],
        "filters": [],
        "complex": []
    }

    query_structures = []

    if not stored_queries_df.empty:
        for _, row in stored_queries_df.iterrows():
            query_text = row['query_text']
            query_id = row['query_id']

            query_lower = query_text.lower()

            if not query_lower.startswith("select"):
                continue

            query_analysis = {
                "query_id": query_id,
                "query_text": query_text,
                "tables_referenced": [],
                "columns_selected": [],
                "has_joins": " join " in query_lower,
                "has_aggregation": any(agg in query_lower for agg in ["count(", "sum(", "avg(", "min(", "max(", "group by"]),
                "has_filters": " where " in query_lower,
                "has_sorting": " order by " in query_lower,
                "complexity": "simple"
            }

            if " from " in query_lower:
                from_part = query_lower[query_lower.find(" from ") + 6:]

                for clause in [" where ", " group by ", " having ", " order by ", " limit "]:
                    if clause in from_part:
                        from_part = from_part[:from_part.find(clause)]

                if " join " in from_part:
                    main_table = from_part[:from_part.find(" join ")].strip()
                    if " as " in main_table:
                        main_table = main_table.split(" as ")[0].strip()
                    elif " " in main_table:
                        main_table = main_table.split(" ")[0].strip()

                    query_analysis["tables_referenced"].append(main_table.strip("()));"))

                    join_parts = from_part.split(" join ")
                    for part in join_parts[1:]:
                        if " on " in part:
                            joined_table = part[:part.find(" on ")].strip()
                        else:
                            joined_table = part.strip()

                        if " as " in joined_table:
                            joined_table = joined_table.split(" as ")[0].strip()
                        elif " " in joined_table:
                            joined_table = joined_table.split(" ")[0].strip()

                        query_analysis["tables_referenced"].append(joined_table.strip("()));"))
                else:
                    tables_part = from_part.strip()
                    for table_ref in tables_part.split(","):
                        table_ref = table_ref.strip()

                        if " as " in table_ref:
                            table_ref = table_ref.split(" as ")[0].strip()
                        elif " " in table_ref:
                            table_ref = table_ref.split(" ")[0].strip()

                        query_analysis["tables_referenced"].append(table_ref.strip("()));"))

            if "select " in query_lower and " from " in query_lower:
                select_part = query_lower[query_lower.find("select ") + 7:query_lower.find(" from ")].strip()

                if select_part == "*":
                    query_analysis["columns_selected"] = ["*"]
                else:
                    columns = []
                    for col in select_part.split(","):
                        col = col.strip()
                        if " as " in col:
                            col = col.split(" as ")[0].strip()
                        columns.append(col)

                    query_analysis["columns_selected"] = columns

            complexity_score = 0
            if query_analysis["has_joins"]:
                complexity_score += len(query_analysis["tables_referenced"]) * 2
            if query_analysis["has_aggregation"]:
                complexity_score += 3
            if query_analysis["has_filters"]:
                complexity_score += 2
            if "subquery" in query_lower or "select " in query_lower[query_lower.find(" from "):]:
                complexity_score += 5
                query_analysis["has_subquery"] = True
            else:
                query_analysis["has_subquery"] = False

            if complexity_score <= 4:
                query_analysis["complexity"] = "simple"
                query_categories["simple_select"].append(query_id)
            elif complexity_score <= 8:
                query_analysis["complexity"] = "moderate"
            else:
                query_analysis["complexity"] = "complex"
                query_categories["complex"].append(query_id)

            if query_analysis["has_joins"]:
                query_categories["joins"].append(query_id)
            if query_analysis["has_aggregation"]:
                query_categories["aggregations"].append(query_id)
            if query_analysis["has_filters"]:
                query_categories["filters"].append(query_id)

            query_structures.append(query_analysis)

    common_joins = {}
    common_selections = {}

    for query in query_structures:
        if len(query["tables_referenced"]) > 1:
            join_key = frozenset(query["tables_referenced"])
            if join_key not in common_joins:
                common_joins[join_key] = []
            common_joins[join_key].append(query["query_id"])

        if "*" not in query["columns_selected"] and query["columns_selected"]:
            selection_key = frozenset(query["columns_selected"])
            if selection_key not in common_selections:
                common_selections[selection_key] = []
            common_selections[selection_key].append(query["query_id"])

    report = f"# Query Pattern Analysis for {database_name}\n\n"
    report += f"## Overview\n\n"
    report += f"This analysis examines {len(query_structures)} queries across {len(tables)} tables to identify patterns "
    report += f"that can inform query optimization and generation.\n\n"

    complexity_counts = {"simple": 0, "moderate": 0, "complex": 0}
    for query in query_structures:
        complexity_counts[query["complexity"]] += 1

    report += "## Query Complexity Distribution\n\n"
    report += f"- Simple queries: {complexity_counts['simple']}\n"
    report += f"- Moderate complexity: {complexity_counts['moderate']}\n"
    report += f"- Complex queries: {complexity_counts['complex']}\n\n"

    report += "## Common Query Types\n\n"
    report += f"- Simple SELECT queries: {len(query_categories['simple_select'])}\n"
    report += f"- Queries with JOINs: {len(query_categories['joins'])}\n"
    report += f"- Queries with aggregations: {len(query_categories['aggregations'])}\n"
    report += f"- Queries with WHERE filters: {len(query_categories['filters'])}\n"
    report += f"- Complex queries: {len(query_categories['complex'])}\n\n"

    if common_joins:
        report += "## Common Table Joins\n\n"
        for tables, queries in sorted(common_joins.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            if len(queries) > 1:
                report += f"### {', '.join(sorted(tables))}\n\n"
                report += f"This join pattern appears in {len(queries)} queries.\n\n"

                example_query_id = queries[0]
                example_query = next((q for q in query_structures if q["query_id"] == example_query_id), None)
                if example_query:
                    query_text = example_query["query_text"]
                    if len(query_text) > 300:
                        query_text = query_text[:300] + "..."

                    report += f"Example query:\n```sql\n{query_text}\n```\n\n"

    if common_selections:
        report += "## Common Column Selections\n\n"
        for columns, queries in sorted(common_selections.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            if len(queries) > 1 and len(columns) > 1:
                report += f"### Columns: {', '.join(sorted(columns))}\n\n"
                report += f"This selection pattern appears in {len(queries)} queries.\n\n"

    report += "## Query Optimization Patterns\n\n"

    select_star_queries = [q for q in query_structures if "*" in q["columns_selected"]]
    large_table_queries = []

    for query in query_structures:
        for table_name in query["tables_referenced"]:
            if table_name in tables and tables[table_name]["row_count"] > 100000:
                large_table_queries.append(query)
                break

    report += "### SELECT * Usage\n\n"
    report += f"Found {len(select_star_queries)} queries using SELECT *. "
    if select_star_queries:
        report += "This pattern retrieves all columns, which can be inefficient for large tables.\n\n"

        example = select_star_queries[0]
        report += f"Example query ID: {example['query_id']}\n"
        query_text = example["query_text"]
        if len(query_text) > 200:
            query_text = query_text[:200] + "..."
        report += f"```sql\n{query_text}\n```\n\n"
    else:
        report += "Good practice to avoid this pattern is being followed.\n\n"

    report += "### Large Table Queries\n\n"
    report += f"Found {len(large_table_queries)} queries accessing tables with over 100,000 rows.\n\n"

    if large_table_queries:
        report += "Tables with high row counts:\n"
        for table_name, info in sorted(tables.items(), key=lambda x: x[1]["row_count"], reverse=True)[:5]:
            if info["row_count"] > 100000:
                report += f"- {table_name}: {info['row_count']:,} rows\n"
        report += "\n"

    report += "## Query Generation Examples\n\n"

    report += "### Simple Query Pattern\n\n"
    simple_example = next((q for q in query_structures if q["complexity"] == "simple"), None)
    if simple_example:
        report += f"```sql\n{simple_example['query_text']}\n```\n\n"

    report += "### Join Query Pattern\n\n"
    join_example = next((q for q in query_structures if q["has_joins"]), None)
    if join_example:
        report += f"```sql\n{join_example['query_text']}\n```\n\n"

    report += "### Aggregation Query Pattern\n\n"
    agg_example = next((q for q in query_structures if q["has_aggregation"]), None)
    if agg_example:
        report += f"```sql\n{agg_example['query_text']}\n```\n\n"

    report_id = f"query_pattern_report_{database_name}_{str(uuid.uuid4())}"
    vector_store.add_texts(
        texts=[report],
        metadatas=[{"type": "query_pattern_report", "database": database_name}],
        ids=[report_id]
    )

    for query in query_structures:
        query_doc = f"# Query: {query['query_id']}\n\n"
        query_doc += f"```sql\n{query['query_text']}\n```\n\n"
        query_doc += f"## Query Properties\n\n"
        query_doc += f"- Complexity: {query['complexity']}\n"
        query_doc += f"- Tables: {', '.join(query['tables_referenced'])}\n"
        query_doc += f"- Has JOINs: {query['has_joins']}\n"
        query_doc += f"- Has aggregations: {query['has_aggregation']}\n"
        query_doc += f"- Has filters: {query['has_filters']}\n"
        query_doc += f"- Has subqueries: {query['has_subquery']}\n"

        vector_store.add_texts(
            texts=[query_doc],
            metadatas=[{
                "type": "query_example",
                "database": database_name,
                "query_id": query['query_id'],
                "complexity": query['complexity'],
                "has_joins": query['has_joins'],
                "has_aggregation": query['has_aggregation'],
                "has_filters": query['has_filters'],
                "has_subquery": query['has_subquery']
            }],
            ids=[f"query_{query['query_id']}"]
        )

    return {
        "query_structures": query_structures,
        "query_categories": query_categories,
        "common_joins": common_joins,
        "common_selections": common_selections,
        "report": report,
        "report_id": report_id
    }

def process_rag_query(user_query, database_name="INSURANCE_DB_10K", k=3):
    print(f"\nProcessing query: {user_query}")
    try:
        # First, get database metadata
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TABLE_NAME, TABLE_TYPE 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
            ORDER BY TABLE_NAME
        """)
        tables = cursor.fetchall()
        print(f"Retrieved metadata for {len(tables)} tables")
        
        # Get column information
        cursor.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """)
        columns = cursor.fetchall()
        print(f"Retrieved metadata for {len(columns)} columns")
        
        # Get primary key information
        cursor.execute("""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE CONSTRAINT_NAME = 'PRIMARY'
            AND TABLE_SCHEMA = CURRENT_SCHEMA()
        """)
        primary_keys = cursor.fetchall()
        
        # Get foreign key information
        cursor.execute("""
            SELECT 
                fk.TABLE_NAME as FK_TABLE,
                fk.COLUMN_NAME as FK_COLUMN,
                pk.TABLE_NAME as PK_TABLE,
                pk.COLUMN_NAME as PK_COLUMN
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk
            JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                ON fk.CONSTRAINT_NAME = rc.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk
                ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
            WHERE fk.TABLE_SCHEMA = CURRENT_SCHEMA()
        """)
        foreign_keys = cursor.fetchall()
        
        cursor.close()

        # Store the metadata in the vector store if not already present
        if not vector_store.get(where={"type": "database_metadata"}):
            metadata_text = f"Database: {database_name}\n\nTables:\n"
            for table in tables:
                metadata_text += f"- {table[0]} ({table[1]})\n"
            
            metadata_text += "\nColumns:\n"
            for column in columns:
                metadata_text += f"- {column[0]}.{column[1]} ({column[2]}, nullable: {column[3]})\n"
            
            metadata_text += "\nPrimary Keys:\n"
            for pk in primary_keys:
                metadata_text += f"- {pk[0]}.{pk[1]}\n"
            
            metadata_text += "\nForeign Keys:\n"
            for fk in foreign_keys:
                metadata_text += f"- {fk[0]}.{fk[1]} -> {fk[2]}.{fk[3]}\n"
            
            vector_store.add_texts(
                texts=[metadata_text],
                metadatas=[{"type": "database_metadata"}],
                ids=["db_metadata"]
            )
            print("Stored database metadata in vector store")

        query_id = store_user_query(user_query)
        print(f"Stored query with ID: {query_id}")
    except Exception as e:
        print(f"Error in metadata retrieval: {e}")
        query_id = str(uuid.uuid4())

    try:
        print("Searching for similar queries...")
        similar_queries = vector_store.similarity_search_with_score(
            query=user_query,
            k=k,
            filter={"type": "original_query"}
        )
        print(f"Found {len(similar_queries)} similar queries")

        print("Searching for relevant tables...")
        relevant_tables = vector_store.similarity_search_with_score(
            query=user_query,
            k=k,
            filter={"type": {"$in": ["table_info", "database_metadata"]}}
        )
        print(f"Found {len(relevant_tables)} relevant tables/structures")

        naming_reports = []
        if "name" in user_query.lower() or "naming" in user_query.lower() or "convention" in user_query.lower():
            naming_reports = vector_store.similarity_search_with_score(
                query=user_query,
                k=2,
                filter={"type": "naming_convention_report"}
            )

        query_patterns = vector_store.similarity_search_with_score(
            query=user_query,
            k=2,
            filter={"type": {"$in": ["query_pattern_report", "query_example"]}}
        )
    except Exception as e:
        return f"Error retrieving relevant information: {str(e)}", query_id

    context = "Database Information:\n"

    if relevant_tables:
        context += "\nRelevant tables:\n"
        for doc, score in relevant_tables:
            table_lines = doc.page_content.split("\n")
            table_name = table_lines[0] if table_lines else "Unknown table"

            context += f"- {table_name}\n"

            columns_section = False
            column_count = 0
            for line in table_lines[1:]:
                if "Columns:" in line:
                    columns_section = True
                    continue
                if columns_section and line.strip().startswith("- ") and column_count < 5:
                    context += f"  {line.strip()}\n"
                    column_count += 1
                if columns_section and "\n\n" in line:
                    break

    if naming_reports:
        context += "\nNaming Conventions:\n"
        for doc, score in naming_reports:
            doc_content = doc.page_content
            if "Column Naming Patterns" in doc_content:
                section = doc_content.split("Column Naming Patterns")[1].split("\n\n")[0]
                context += f"{section}\n"
            else:
                context += "Database uses a mixture of naming conventions. "
                if "snake_case" in doc_content:
                    context += "snake_case is commonly used for column names.\n"

    if similar_queries:
        context += "\nSimilar queries:\n"
        for i, (doc, score) in enumerate(similar_queries[:2], 1):
            query_text = doc.page_content
            context += f"{i}. {query_text}\n"

    prompt_template = """
    You are a database expert assistant specializing in SQL and database design best practices.

    Here is the current database structure:
    {context}

    User query: {query}

    Please provide a detailed analysis with specific recommendations for the actual database structure shown above.
    For each recommendation, include:
    1. The exact table(s) and column(s) affected
    2. Current state of the affected objects
    3. Specific changes needed
    4. Expected benefits of the changes

    Focus on:
    - Identifying redundant tables or columns by name
    - Pointing out specific naming convention issues
    - Suggesting specific performance improvements
    - Identifying missing relationships between specific tables

    Response:
    """

    formatted_prompt = prompt_template.format(context=context, query=user_query)

    # Return the context and prompt for your new model interface to use
    return {
        "query_id": query_id,
        "context": context,
        "prompt": formatted_prompt
    }

# Initialize connections
conn, vector_store = setup_connections()

app = Flask(__name__)
print("test 2: flask created")
CORS(app)

@app.route("/")
def index():
    return send_file("frontend_rag.html")

@app.route("/rag_query", methods=["POST"])
def rag_query_endpoint():
    payload = request.get_json()
    question = payload.get("query", "")
    result = process_rag_query(question)
    
    # Generate response using OpenAI's GPT-3.5-turbo
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful database expert assistant."},
            {"role": "user", "content": result["prompt"]}
        ],
        temperature=0.2,
        max_tokens=256,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.1
    )
    
    answer = response.choices[0].message.content.strip()
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)