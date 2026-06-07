# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ... [License text omitted for brevity] ...

import datetime
from zoneinfo import ZoneInfo
import os
import pandas as pd

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from dotenv import load_dotenv

load_dotenv()




# ---------------------------------------------------------
# AI STUDIO CONFIGURATION
# Set your API key here (or export it in your terminal via `export GEMINI_API_KEY="..."`)
os.environ["GEMINI_API_KEY"] = "YOUR_AI_STUDIO_API_KEY_HERE"
# ---------------------------------------------------------


def get_weather(query: str) -> str:
    """Simulates a web search. Use it get information on weather.

    Args:
        query: A string containing the location to get weather information for.

    Returns:
        A string with the simulated weather information for the queried location.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

def get_current_time(query: str) -> str:
    """Simulates getting the current time for a city.

    Args:
        city: The name of the city to get the current time for.

    Returns:
        A string with the current time information.
    """
    if "sf" in query.lower() or "san francisco" in query.lower():
        tz_identifier = "America/Los_Angeles"
    else:
        return f"Sorry, I don't have timezone information for query: {query}."

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return f"The current time for query {query} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"

def read_dataset(file_path: str) -> str:
    """Reads a local dataset and returns its raw content for analysis.

    Args:
        file_path: The local path to the dataset file (CSV or Excel).

    Returns:
        The content of the dataset as a CSV string.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found. Please ensure the path is correct."

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return "Error: Unsupported file format. Please provide a .csv or .xlsx file."
        
        # Convert the dataframe to a CSV string to pass to the LLM.
        return df.to_csv(index=False)
    except Exception as e:
        return f"Error reading the dataset: {str(e)}"

root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are Agent 1, a data analysis specialist. Your task is to perform a thorough analysis to find duplicates, anomalies, and suspicious patterns."
    ),
    tools=[],
)

app = App(
    root_agent=root_agent,
    name="app",
)