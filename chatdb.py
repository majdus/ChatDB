import argparse
import os

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_ollama.llms import OllamaLLM as Ollama


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--db", required=True, help="Path to the database file.")
        parser.add_argument("--model", required=True, help="The model to use.")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output.")

        args = parser.parse_args()

        db_path = args.db
        model = args.model
        verbose = args.verbose

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

        llm = Ollama(model=model)
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

        db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=verbose)

        while True:
            request = input("Put your request here (or 'exit') : ")
            if request.lower() == "exit":
                break
            try:
                response = db_chain.invoke(input={"query": request})
                print(response)
                continue
            except Exception as e:
                print(f"An error occurred : {e}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
